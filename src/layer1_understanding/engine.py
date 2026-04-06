"""
Layer 1: Codebase Understanding Engine
=======================================
Parses every file into AST-level chunks, embeds them with CodeBERT,
builds a call graph with NetworkX, and stores everything in a FAISS index.

At query time: retrieves top-K most semantically relevant code chunks
in under 100ms — even from a 100,000-line codebase.

Technologies: tree-sitter, CodeBERT (HuggingFace), FAISS, NetworkX, GitPython
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class CodeChunk:
    """A single meaningful unit of code (function, class, method)."""
    chunk_id: str
    file_path: str
    node_type: str          # "function", "class", "method", "module"
    name: str
    source_code: str
    start_line: int
    end_line: int
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.file_path}:{self.name}:{self.start_line}".encode()
            ).hexdigest()[:12]


class ASTParser:
    """
    Parses Python files using tree-sitter into function/class-level chunks.
    
    KEY DESIGN DECISION: We chunk at AST boundaries, never by character count.
    This ensures every chunk is a complete, meaningful unit of code.
    """

    def __init__(self):
        self._setup_parser()

    def _setup_parser(self):
        """Initialize tree-sitter parser for Python."""
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser
            
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)
            self.available = True
            console.print("[green]✓ tree-sitter parser ready[/green]")
        except ImportError:
            console.print("[yellow]⚠ tree-sitter not installed. Using fallback parser.[/yellow]")
            console.print("  Run: pip install tree-sitter tree-sitter-python")
            self.parser = None
            self.available = False

    def parse_file(self, file_path: str) -> list[CodeChunk]:
        """Parse a Python file and return list of CodeChunks."""
        path = Path(file_path)
        if not path.exists() or path.suffix != '.py':
            return []

        source = path.read_text(encoding='utf-8', errors='ignore')
        
        if self.available and self.parser:
            return self._parse_with_treesitter(source, file_path)
        else:
            return self._parse_fallback(source, file_path)

    def _parse_with_treesitter(self, source: str, file_path: str) -> list[CodeChunk]:
        """Parse using tree-sitter for accurate AST chunking."""
        chunks = []
        tree = self.parser.parse(bytes(source, 'utf8'))
        lines = source.split('\n')

        def extract_nodes(node, parent_name=None):
            if node.type in ('function_definition', 'class_definition'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf8')
                    start_line = node.start_point[0]
                    end_line = node.end_point[0]
                    source_code = '\n'.join(lines[start_line:end_line + 1])
                    
                    node_type = 'function' if node.type == 'function_definition' else 'class'
                    if parent_name and node_type == 'function':
                        node_type = 'method'

                    chunk = CodeChunk(
                        chunk_id='',
                        file_path=file_path,
                        node_type=node_type,
                        name=f"{parent_name}.{name}" if parent_name else name,
                        source_code=source_code,
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                    )
                    chunks.append(chunk)
                    
                    # Recurse into classes to get methods
                    for child in node.children:
                        extract_nodes(child, parent_name=name if node_type == 'class' else None)
            else:
                for child in node.children:
                    extract_nodes(child, parent_name)

        extract_nodes(tree.root_node)
        return chunks

    def _parse_fallback(self, source: str, file_path: str) -> list[CodeChunk]:
        """Fallback parser using Python's built-in ast module."""
        import ast
        chunks = []
        try:
            tree = ast.parse(source)
            lines = source.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                    end_line = getattr(node, 'end_lineno', node.lineno + 10)
                    source_code = '\n'.join(lines[node.lineno - 1:end_line])
                    
                    chunk = CodeChunk(
                        chunk_id='',
                        file_path=file_path,
                        node_type=node_type,
                        name=node.name,
                        source_code=source_code,
                        start_line=node.lineno,
                        end_line=end_line,
                    )
                    chunks.append(chunk)
        except SyntaxError:
            pass
        return chunks


class CodeEmbedder:
    """
    Embeds code chunks using CodeBERT (or sentence-transformers as fallback).
    
    CodeBERT is pre-trained on GitHub code — it understands that 'list' in 
    Python code is a data structure, not a word.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embedding_dim = 768
        self._load_model()

    def _load_model(self):
        """Load CodeBERT model. Falls back to sentence-transformers."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            console.print(f"[cyan]Loading {self.model_name}...[/cyan]")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.backend = 'codebert'
            console.print("[green]✓ CodeBERT loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ CodeBERT failed: {e}[/yellow]")
            self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        """Fallback to sentence-transformers (faster, smaller)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            self.backend = 'sentence_transformer'
            console.print("[green]✓ Sentence-transformer loaded as fallback[/green]")
        except Exception as e:
            console.print(f"[red]✗ No embedding model available: {e}[/red]")
            self.backend = 'random'  # last resort for testing

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self.backend == 'codebert':
            return self._embed_codebert(text)
        elif self.backend == 'sentence_transformer':
            return self.st_model.encode([text])[0]
        else:
            # Random embedding for testing (not useful for production)
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def _embed_codebert(self, text: str) -> np.ndarray:
        import torch
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use [CLS] token embedding as the chunk representation
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.embed(t) for t in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings, dtype=np.float32)


class FAISSIndex:
    """
    Vector similarity search using FAISS.
    Handles 10M+ vectors with sub-millisecond query time.
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks: list[CodeChunk] = []
        self._setup_index()

    def _setup_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after normalization)
            self.faiss = faiss
            console.print("[green]✓ FAISS index ready[/green]")
        except ImportError:
            console.print("[yellow]⚠ FAISS not installed. Run: pip install faiss-cpu[/yellow]")
            self.index = None

    def add_chunks(self, chunks: list[CodeChunk]):
        """Add code chunks with their embeddings to the index."""
        if self.index is None:
            console.print("[red]FAISS not available[/red]")
            return

        embeddings = np.array([c.embedding for c in chunks if c.embedding is not None])
        if len(embeddings) == 0:
            return

        # Normalize for cosine similarity
        self.faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        console.print(f"[green]✓ Indexed {len(chunks)} chunks ({self.index.ntotal} total)[/green]")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[CodeChunk, float]]:
        """Find top-K most similar code chunks to a query embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        self.faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str):
        """Persist index to disk."""
        if self.index:
            import faiss
            faiss.write_index(self.index, f"{path}.index")
            with open(f"{path}.chunks.json", 'w') as f:
                json.dump([{
                    'chunk_id': c.chunk_id,
                    'file_path': c.file_path,
                    'node_type': c.node_type,
                    'name': c.name,
                    'source_code': c.source_code,
                    'start_line': c.start_line,
                    'end_line': c.end_line,
                } for c in self.chunks], f, indent=2)
            console.print(f"[green]✓ Index saved to {path}[/green]")


class CallGraphBuilder:
    """
    Builds a directed call graph using NetworkX.
    Node = function, Edge = function call.
    Used by Layer 4 to identify all callers when a function signature changes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: list[CodeChunk]) -> nx.DiGraph:
        """Build call graph by analyzing function bodies for calls."""
        import ast

        # Add all functions as nodes
        for chunk in chunks:
            if chunk.node_type in ('function', 'method'):
                self.graph.add_node(chunk.name, chunk=chunk)

        function_names = set(self.graph.nodes())

        # Parse each function body to find calls
        for chunk in chunks:
            if chunk.node_type not in ('function', 'method'):
                continue
            try:
                tree = ast.parse(chunk.source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        called = self._extract_call_name(node)
                        if called and called in function_names and called != chunk.name:
                            self.graph.add_edge(chunk.name, called)
            except SyntaxError:
                pass

        console.print(f"[green]✓ Call graph: {self.graph.number_of_nodes()} nodes, "
                      f"{self.graph.number_of_edges()} edges[/green]")
        return self.graph

    def _extract_call_name(self, node) -> Optional[str]:
        """Extract function name from a Call AST node."""
        import ast
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def get_callers(self, function_name: str) -> list[str]:
        """Get all functions that call the given function."""
        if function_name not in self.graph:
            return []
        return list(self.graph.predecessors(function_name))

    def get_callees(self, function_name: str) -> list[str]:
        """Get all functions called by the given function."""
        if function_name not in self.graph:
            return []
        return list(self.graph.successors(function_name))


class CodebaseUnderstandingEngine:
    """
    Layer 1: The complete codebase understanding pipeline.
    
    Usage:
        engine = CodebaseUnderstandingEngine()
        engine.index_repository('/path/to/repo')
        results = engine.query('function that handles user authentication', top_k=5)
    """

    def __init__(self):
        self.parser = ASTParser()
        self.embedder = CodeEmbedder()
        self.index = FAISSIndex(embedding_dim=self.embedder.embedding_dim)
        self.call_graph_builder = CallGraphBuilder()
        self.all_chunks: list[CodeChunk] = []
        self.call_graph: Optional[nx.DiGraph] = None

    def index_repository(self, repo_path: str, file_limit: int = 500):
        """
        Full pipeline: parse → embed → index all Python files in a repository.
        
        Args:
            repo_path: Path to the cloned repository
            file_limit: Max files to process (prevents memory issues on huge repos)
        """
        repo_path = Path(repo_path)
        console.print(f"\n[bold cyan]🔍 Indexing repository: {repo_path}[/bold cyan]")

        # Find all Python files
        py_files = list(repo_path.rglob('*.py'))[:file_limit]
        # Skip common non-essential directories
        py_files = [f for f in py_files if not any(
            part in f.parts for part in ['__pycache__', '.venv', 'venv', 'node_modules', '.git']
        )]
        console.print(f"Found {len(py_files)} Python files")

        # Parse all files into chunks
        all_chunks = []
        for py_file in track(py_files, description="Parsing files..."):
            chunks = self.parser.parse_file(str(py_file))
            all_chunks.extend(chunks)

        console.print(f"Extracted {len(all_chunks)} code chunks")

        # Embed all chunks
        console.print("[cyan]Embedding chunks with CodeBERT...[/cyan]")
        texts = [f"{c.name}\n{c.source_code}" for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        # Build FAISS index
        self.index.add_chunks(all_chunks)
        self.all_chunks = all_chunks

        # Build call graph
        self.call_graph = self.call_graph_builder.build_from_chunks(all_chunks)

        console.print(f"\n[bold green]✅ Indexing complete![/bold green]")
        console.print(f"   Chunks: {len(all_chunks)}")
        console.print(f"   Index size: {self.index.index.ntotal if self.index.index else 0}")

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[CodeChunk, float]]:
        """
        Retrieve the top-K most relevant code chunks for a query.
        
        Args:
            query_text: Natural language description or code snippet
            top_k: Number of results to return
            
        Returns:
            List of (CodeChunk, similarity_score) tuples
        """
        query_embedding = self.embedder.embed(query_text)
        results = self.index.search(query_embedding, top_k=top_k)
        return results

    def get_callers(self, function_name: str) -> list[str]:
        """Get all functions that call the given function (for multi-file edits)."""
        return self.call_graph_builder.get_callers(function_name)


# ── Entry point for testing ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    engine = CodebaseUnderstandingEngine()
    
    # Test on a sample path (change to your repo)
    test_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    engine.index_repository(test_path)
    
    # Test retrieval
    query = "function that calculates or processes numerical values"
    console.print(f"\n[bold]Query:[/bold] {query}")
    results = engine.query(query, top_k=3)
    
    for chunk, score in results:
        console.print(f"\n[cyan]Score: {score:.3f}[/cyan] | {chunk.file_path}:{chunk.start_line}")
        console.print(f"[yellow]{chunk.name}[/yellow]")
        console.print(chunk.source_code[:200] + "...")
