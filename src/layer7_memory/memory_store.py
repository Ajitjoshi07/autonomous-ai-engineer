"""
Layer 7: Long-Term Memory & Self-Improvement
=============================================
Stores structured episodic memories for every task. Retrieves similar
past experiences at planning time to bias the agent toward strategies
that have worked before.

This is the feature that makes the system measurably improve over time —
the agent's resolve rate and iteration efficiency improve the more tasks it completes.

Technologies: FAISS memory index, sentence-transformers, JSON structured logs
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    A complete episodic memory of one agent task session.
    Stored after EVERY task — successful or not.
    """
    # Identity
    memory_id: str
    timestamp: str
    repository: str

    # Task info
    issue_text: str
    issue_embedding: Optional[list] = None   # Stored as list for JSON serialization

    # Outcome
    success: bool = False
    iterations_required: int = 0
    total_time_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    # Strategy that worked (or was tried)
    root_cause_classification: str = ""
    fix_strategy_used: str = ""
    target_files: list[str] = field(default_factory=list)
    target_functions: list[str] = field(default_factory=list)

    # Quality
    critic_approved: bool = False
    critic_score: float = 0.0

    # What failed (for learning from failures)
    failed_strategies: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    # PR info (if created)
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'repository': self.repository,
            'issue_text': self.issue_text[:500],
            'issue_embedding': self.issue_embedding,
            'success': self.success,
            'iterations_required': self.iterations_required,
            'total_time_seconds': self.total_time_seconds,
            'estimated_cost_usd': self.estimated_cost_usd,
            'root_cause_classification': self.root_cause_classification,
            'fix_strategy_used': self.fix_strategy_used,
            'target_files': self.target_files,
            'target_functions': self.target_functions,
            'critic_approved': self.critic_approved,
            'critic_score': self.critic_score,
            'failed_strategies': self.failed_strategies,
            'failure_reasons': self.failure_reasons,
            'pr_url': self.pr_url,
            'pr_number': self.pr_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryRecord':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemorySearchResult:
    """A retrieved memory with its similarity score."""
    record: MemoryRecord
    similarity_score: float

    def to_prompt_text(self) -> str:
        """Format for injection into the planning prompt."""
        status = "✅ RESOLVED" if self.record.success else "❌ FAILED"
        return (
            f"[Past Issue — Similarity: {self.similarity_score:.2f}] {status}\n"
            f"Repository: {self.record.repository}\n"
            f"Root Cause: {self.record.root_cause_classification}\n"
            f"Strategy Used: {self.record.fix_strategy_used}\n"
            f"Iterations Needed: {self.record.iterations_required}\n"
            f"Critic Score: {self.record.critic_score:.2f}\n"
            + (f"Failed Approaches: {', '.join(self.record.failed_strategies)}\n"
               if self.record.failed_strategies else "")
        )


# ── Memory Store ─────────────────────────────────────────────────────────────

class EpisodicMemoryStore:
    """
    Persistent vector-based episodic memory store.

    Stores and retrieves memories using FAISS for fast similarity search.
    Falls back to linear scan if FAISS is unavailable.

    Usage:
        memory = EpisodicMemoryStore()
        memory.store(record)
        similar = memory.retrieve_similar(issue_text, top_k=3)
    """

    SIMILARITY_THRESHOLD = 0.82  # Only return memories above this similarity

    def __init__(self, store_path: str = None):
        self.store_path = Path(
            store_path or os.getenv('MEMORY_STORE_PATH', './data/memory_store')
        )
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.records: list[MemoryRecord] = []
        self.embeddings: list[np.ndarray] = []
        self.embedder = self._init_embedder()
        self.faiss_index = None
        self.embedding_dim = 384

        self._load_from_disk()
        self._rebuild_faiss_index()

    def _init_embedder(self):
        """Initialize sentence-transformer for issue embedding."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[green]✓ Memory embedder ready (sentence-transformers)[/green]")
            return model
        except ImportError:
            console.print("[yellow]⚠ sentence-transformers not installed[/yellow]")
            console.print("  Run: pip install sentence-transformers")
            return None

    def store(self, record: MemoryRecord) -> bool:
        """
        Store a new memory record.
        Called after EVERY task completion (success or failure).
        """
        console.print(f"\n[cyan]💾 Storing memory: {record.memory_id}[/cyan]")

        # Embed the issue text
        if self.embedder:
            embedding = self.embedder.encode([record.issue_text[:512]])[0]
            record.issue_embedding = embedding.tolist()
            self.embeddings.append(embedding)
        else:
            self.embeddings.append(np.random.randn(self.embedding_dim).astype(np.float32))

        self.records.append(record)

        # Rebuild FAISS index with new record
        self._rebuild_faiss_index()

        # Persist to disk
        self._save_record(record)

        status = "✅ success" if record.success else "❌ failed"
        console.print(
            f"[green]Memory stored ({status}, {record.iterations_required} iterations, "
            f"score: {record.critic_score:.2f})[/green]"
        )
        return True

    def retrieve_similar(
        self,
        issue_text: str,
        top_k: int = 3,
        min_similarity: float = None
    ) -> list[MemorySearchResult]:
        """
        Retrieve memories of similar past issues.

        Args:
            issue_text: Current issue to find similar memories for
            top_k: Maximum number of memories to return
            min_similarity: Minimum cosine similarity (default: SIMILARITY_THRESHOLD)

        Returns:
            List of MemorySearchResult sorted by similarity (highest first)
        """
        if not self.records:
            return []

        threshold = min_similarity or self.SIMILARITY_THRESHOLD

        if self.embedder is None:
            return []

        # Embed the query
        query_embedding = self.embedder.encode([issue_text[:512]])[0]
        query_embedding = query_embedding.astype(np.float32)

        # Search
        if self.faiss_index and self.faiss_index.ntotal > 0:
            results = self._faiss_search(query_embedding, top_k)
        else:
            results = self._linear_search(query_embedding, top_k)

        # Filter by threshold
        filtered = [r for r in results if r.similarity_score >= threshold]

        if filtered:
            console.print(
                f"[green]🧠 Memory: found {len(filtered)} similar past issue(s) "
                f"(similarity ≥ {threshold})[/green]"
            )
        else:
            console.print(f"[dim]Memory: no similar past issues found[/dim]")

        return filtered

    def _faiss_search(self, query: np.ndarray, top_k: int) -> list[MemorySearchResult]:
        """Search using FAISS (fast)."""
        import faiss

        query_norm = query.copy()
        faiss.normalize_L2(query_norm.reshape(1, -1))

        scores, indices = self.faiss_index.search(
            query_norm.reshape(1, -1),
            min(top_k, self.faiss_index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.records):
                results.append(MemorySearchResult(
                    record=self.records[idx],
                    similarity_score=float(score)
                ))
        return results

    def _linear_search(self, query: np.ndarray, top_k: int) -> list[MemorySearchResult]:
        """Fallback linear scan when FAISS unavailable."""
        if not self.embeddings:
            return []

        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        scores = []

        for i, emb in enumerate(self.embeddings):
            emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
            score = float(np.dot(query_norm, emb_norm))
            scores.append((score, i))

        scores.sort(reverse=True)
        return [
            MemorySearchResult(record=self.records[i], similarity_score=s)
            for s, i in scores[:top_k]
        ]

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from all stored embeddings."""
        if not self.embeddings:
            return
        try:
            import faiss
            index = faiss.IndexFlatIP(self.embedding_dim)
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            self.faiss_index = index
        except ImportError:
            pass

    def _save_record(self, record: MemoryRecord):
        """Persist a single record to disk as JSON."""
        file_path = self.store_path / f"{record.memory_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, indent=2)

    def _load_from_disk(self):
        """Load all memory records from disk on startup."""
        json_files = list(self.store_path.glob('*.json'))
        if not json_files:
            return

        console.print(f"[cyan]Loading {len(json_files)} memories from disk...[/cyan]")
        loaded = 0
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                record = MemoryRecord.from_dict(data)
                self.records.append(record)

                # Rebuild embedding from stored data
                if record.issue_embedding:
                    self.embeddings.append(np.array(record.issue_embedding, dtype=np.float32))
                else:
                    self.embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                loaded += 1
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load {json_file.name}: {e}[/yellow]")

        console.print(f"[green]✓ Loaded {loaded} memories[/green]")

    def get_statistics(self) -> dict:
        """Get aggregate statistics about stored memories."""
        if not self.records:
            return {'total': 0}

        successful = [r for r in self.records if r.success]
        return {
            'total': len(self.records),
            'successful': len(successful),
            'failed': len(self.records) - len(successful),
            'resolve_rate': len(successful) / len(self.records),
            'avg_iterations': sum(r.iterations_required for r in self.records) / len(self.records),
            'avg_cost_usd': sum(r.estimated_cost_usd for r in self.records) / len(self.records),
            'avg_critic_score': sum(r.critic_score for r in successful) / max(len(successful), 1),
        }

    def print_statistics(self):
        """Display a rich statistics table."""
        stats = self.get_statistics()

        table = Table(title="Memory Store Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total Tasks", str(stats.get('total', 0)))
        table.add_row("Successful", f"[green]{stats.get('successful', 0)}[/green]")
        table.add_row("Failed", f"[red]{stats.get('failed', 0)}[/red]")
        resolve_rate = stats.get('resolve_rate', 0)
        table.add_row("Resolve Rate", f"{resolve_rate:.1%}")
        table.add_row("Avg Iterations", f"{stats.get('avg_iterations', 0):.1f}")
        table.add_row("Avg Cost/Task", f"${stats.get('avg_cost_usd', 0):.3f}")
        table.add_row("Avg Critic Score", f"{stats.get('avg_critic_score', 0):.2f}")

        console.print(table)


# ── Helper: Build Memory Record from Session ──────────────────────────────────

def build_memory_record(
    repository: str,
    issue_text: str,
    plan,
    test_results: list,
    critic_feedback,
    success: bool,
    start_time: float,
    pr_url: str = None,
    pr_number: int = None
) -> MemoryRecord:
    """
    Build a MemoryRecord from all the data collected during a task session.
    Called by the main orchestrator after each task.
    """
    memory_id = hashlib.md5(
        f"{repository}:{issue_text[:100]}:{time.time()}".encode()
    ).hexdigest()[:12]

    # Collect failed strategies from test history
    failed_strategies = []
    failure_reasons = []
    for i, result in enumerate(test_results[:-1] if success else test_results):
        if result.failed > 0 and result.failures:
            failed_strategies.append(f"iteration_{i+1}")
            failure_reasons.append(result.failures[0].assertion_message[:100]
                                   if result.failures else "unknown")

    return MemoryRecord(
        memory_id=memory_id,
        timestamp=datetime.utcnow().isoformat(),
        repository=repository,
        issue_text=issue_text,
        success=success,
        iterations_required=len(test_results),
        total_time_seconds=time.time() - start_time,
        root_cause_classification=plan.root_cause if plan else "",
        fix_strategy_used=plan.selected_strategy.approach if plan else "",
        target_files=plan.target_files if plan else [],
        target_functions=plan.target_functions if plan else [],
        critic_approved=critic_feedback.approved if critic_feedback else False,
        critic_score=critic_feedback.overall_score if critic_feedback else 0.0,
        failed_strategies=failed_strategies,
        failure_reasons=failure_reasons,
        pr_url=pr_url,
        pr_number=pr_number,
    )


# ── Entry point for testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    memory = EpisodicMemoryStore(store_path='/tmp/test_memory')

    # Store a test record
    record = MemoryRecord(
        memory_id='test001',
        timestamp=datetime.utcnow().isoformat(),
        repository='test/repo',
        issue_text='calculate_discount returns wrong value when rate is 0',
        success=True,
        iterations_required=2,
        root_cause_classification='incorrect conditional in discount function',
        fix_strategy_used='fix arithmetic formula: price * (1 - rate)',
        target_files=['calculator.py'],
        target_functions=['calculate_discount'],
        critic_approved=True,
        critic_score=0.91,
    )
    memory.store(record)

    # Retrieve similar
    results = memory.retrieve_similar(
        "function returns 0 when input is zero instead of original value",
        top_k=3,
        min_similarity=0.3  # Lower threshold for testing
    )

    console.print(f"\nRetrieved {len(results)} similar memories:")
    for r in results:
        console.print(r.to_prompt_text())

    memory.print_statistics()
