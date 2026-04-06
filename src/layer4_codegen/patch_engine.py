"""
Layer 4: Code Generation & Patch Engine
==========================================
Generates unified diffs against target files, applies them with conflict
detection and retry, and coordinates multi-file edits using the call graph
from Layer 1.

KEY PRINCIPLE: Generate patches (diffs), never full file rewrites.
This prevents introducing regressions in untouched code.

Technologies: difflib, unidiff, GPT-4o function calling, AST-aware patching
"""

import os
import json
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.syntax import Syntax

console = Console()


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class FilePatch:
    """A single file's worth of changes in unified diff format."""
    file_path: str           # Relative path inside the repo
    original_content: str
    patched_content: str
    diff_text: str           # The unified diff string
    lines_added: int = 0
    lines_removed: int = 0

    def __post_init__(self):
        added = sum(1 for l in self.diff_text.split('\n') if l.startswith('+') and not l.startswith('+++'))
        removed = sum(1 for l in self.diff_text.split('\n') if l.startswith('-') and not l.startswith('---'))
        self.lines_added = added
        self.lines_removed = removed


@dataclass
class MultiFilePatch:
    """A coordinated patch across one or more files."""
    patches: list[FilePatch] = field(default_factory=list)
    description: str = ""

    @property
    def total_files(self) -> int:
        return len(self.patches)

    @property
    def total_lines_changed(self) -> int:
        return sum(p.lines_added + p.lines_removed for p in self.patches)

    def summary(self) -> str:
        return (f"{self.total_files} file(s) changed, "
                f"+{sum(p.lines_added for p in self.patches)} "
                f"-{sum(p.lines_removed for p in self.patches)}")


# ── Prompts ──────────────────────────────────────────────────────────────────

CODE_GEN_PROMPT = """You are an expert software engineer implementing a precise bug fix.

## Repair Plan
{plan_summary}

## Target File: {file_path}
```python
{file_content}
```

## Additional Context (related functions)
{context_chunks}

## Your Task
Generate a minimal, correct fix. Rules:
1. Change ONLY what is necessary to fix the bug
2. Do NOT refactor unrelated code
3. Do NOT change function signatures unless the plan requires it
4. Preserve all existing comments and docstrings

Return ONLY valid JSON:
{{
  "reasoning": "Step-by-step explanation of what you're changing and why",
  "patched_content": "The complete corrected file content"
}}"""

RETRY_PROMPT = """You are fixing a bug. Your previous patch failed tests.

## Original Plan
{plan_summary}

## Current File Content (after your last patch)
```python
{current_content}
```

## Test Failures You Must Fix
{failure_details}

## What NOT to do
- Do NOT repeat the same approach that caused these failures
- Do NOT change unrelated code
- Analyze EACH failure traceback carefully before writing any code

Return ONLY valid JSON:
{{
  "reasoning": "What went wrong and how you're fixing it differently",
  "patched_content": "The complete corrected file content"
}}"""


# ── Code Generator ────────────────────────────────────────────────────────────

class CodeGenerator:
    """
    Layer 4: Generates and applies code patches.

    Usage:
        generator = CodeGenerator()
        patch = generator.generate_patch(plan, retrieved_chunks, repo_path)
        success = generator.apply_patch(patch, repo_path)
    """

    def __init__(self):
        self.llm = self._init_llm()
        self.patch_history: list[MultiFilePatch] = []

    def _init_llm(self):
        """Initialize LLM client (reuses Layer 2's client)."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from layer2_planning.agent import LLMClient
            return LLMClient()
        except Exception as e:
            console.print(f"[yellow]⚠ LLM client init issue: {e}[/yellow]")
            return None

    def generate_patch(
        self,
        plan,
        retrieved_chunks: list[dict],
        repo_path: str,
        sandbox=None
    ) -> Optional[MultiFilePatch]:
        """
        Generate a code patch based on the repair plan.

        Args:
            plan: RepairPlan from Layer 2
            retrieved_chunks: Relevant code chunks from Layer 1
            repo_path: Path to the repository root
            sandbox: Optional Layer 3 sandbox (reads files from inside container)

        Returns:
            MultiFilePatch with all file changes, or None on failure
        """
        console.print(f"\n[bold cyan]⚙ Code Generator: Generating patch...[/bold cyan]")

        patches = []
        target_files = plan.target_files if hasattr(plan, 'target_files') else []

        if not target_files:
            console.print("[yellow]⚠ No target files in plan[/yellow]")
            return None

        for file_path in target_files:
            patch = self._generate_single_file_patch(
                file_path=file_path,
                plan=plan,
                retrieved_chunks=retrieved_chunks,
                repo_path=repo_path,
                sandbox=sandbox
            )
            if patch:
                patches.append(patch)

        if not patches:
            return None

        multi_patch = MultiFilePatch(
            patches=patches,
            description=f"Fix: {plan.issue_summary if hasattr(plan, 'issue_summary') else 'bug fix'}"
        )

        console.print(f"[green]✓ Patch generated: {multi_patch.summary()}[/green]")
        self._display_patch(multi_patch)
        self.patch_history.append(multi_patch)
        return multi_patch

    def _generate_single_file_patch(
        self,
        file_path: str,
        plan,
        retrieved_chunks: list[dict],
        repo_path: str,
        sandbox=None
    ) -> Optional[FilePatch]:
        """Generate a patch for a single file."""
        # Read the current file content
        original_content = self._read_file(file_path, repo_path, sandbox)
        if original_content is None:
            console.print(f"[red]✗ Cannot read {file_path}[/red]")
            return None

        # Build context from retrieved chunks (exclude the target file itself)
        context_chunks = [
            c for c in retrieved_chunks
            if c.get('file_path', '') != file_path
        ]
        context_text = self._format_context(context_chunks[:3])

        # Build plan summary
        plan_summary = self._format_plan(plan)

        prompt = CODE_GEN_PROMPT.format(
            plan_summary=plan_summary,
            file_path=file_path,
            file_content=original_content[:4000],  # Limit context
            context_chunks=context_text
        )

        if not self.llm:
            console.print("[red]✗ LLM not available[/red]")
            return None

        try:
            response = self.llm.chat(
                system_prompt=(
                    "You are an expert software engineer. "
                    "Return only valid JSON with 'reasoning' and 'patched_content' keys."
                ),
                user_prompt=prompt,
                temperature=0.1  # Low temperature for precise code generation
            )

            data = json.loads(response)
            patched_content = data.get('patched_content', '')
            reasoning = data.get('reasoning', '')

            if not patched_content:
                console.print("[red]✗ LLM returned empty patch[/red]")
                return None

            console.print(f"[cyan]Reasoning: {reasoning[:150]}...[/cyan]")

            # Generate the unified diff
            diff_text = self._compute_diff(
                original=original_content,
                patched=patched_content,
                file_path=file_path
            )

            return FilePatch(
                file_path=file_path,
                original_content=original_content,
                patched_content=patched_content,
                diff_text=diff_text
            )

        except json.JSONDecodeError as e:
            console.print(f"[red]✗ JSON parse error: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]✗ Code generation error: {e}[/red]")
            return None

    def retry_with_feedback(
        self,
        plan,
        failures: list,
        retry_prompt: str,
        iteration: int
    ) -> Optional[MultiFilePatch]:
        """
        Generate a revised patch based on test failure feedback.
        Called by Layer 5's feedback loop.

        Args:
            plan: Original RepairPlan
            failures: List of TestFailure objects from Layer 5
            retry_prompt: Formatted failure context from Layer 5
            iteration: Current iteration number (uses cheaper model for iter 2+)

        Returns:
            New MultiFilePatch or None
        """
        console.print(f"[cyan]🔁 Retry {iteration}: Regenerating patch with failure context...[/cyan]")

        patches = []
        for patch_history in self.patch_history[-1:]:  # Use most recent patch
            for file_patch in patch_history.patches:
                failure_details = '\n\n'.join(f.to_prompt_text() for f in failures[:3])

                prompt = RETRY_PROMPT.format(
                    plan_summary=self._format_plan(plan),
                    current_content=file_patch.patched_content[:4000],
                    failure_details=failure_details
                )

                if not self.llm:
                    return None

                try:
                    response = self.llm.chat(
                        system_prompt=(
                            "You are debugging code. Analyze the test failures carefully. "
                            "Return only valid JSON with 'reasoning' and 'patched_content' keys."
                        ),
                        user_prompt=prompt,
                        temperature=0.05  # Very low — be precise
                    )

                    data = json.loads(response)
                    patched_content = data.get('patched_content', '')

                    if patched_content:
                        diff = self._compute_diff(
                            original=file_patch.original_content,
                            patched=patched_content,
                            file_path=file_patch.file_path
                        )
                        patches.append(FilePatch(
                            file_path=file_patch.file_path,
                            original_content=file_patch.original_content,
                            patched_content=patched_content,
                            diff_text=diff
                        ))

                except Exception as e:
                    console.print(f"[red]✗ Retry generation failed: {e}[/red]")

        if patches:
            new_patch = MultiFilePatch(patches=patches, description=f"Retry {iteration}")
            self.patch_history.append(new_patch)
            return new_patch
        return None

    def apply_patch(
        self,
        patch: MultiFilePatch,
        repo_path: str = None,
        sandbox=None
    ) -> bool:
        """
        Write the patched file contents to disk (or into the sandbox).

        Args:
            patch: MultiFilePatch to apply
            repo_path: Local repo path (if writing to disk)
            sandbox: Layer 3 sandbox (if writing into container)

        Returns:
            True if all patches applied successfully
        """
        console.print(f"[cyan]Applying patch: {patch.summary()}[/cyan]")
        success = True

        for file_patch in patch.patches:
            try:
                if sandbox and sandbox.container:
                    # Write into Docker container
                    container_path = f"/workspace/{file_patch.file_path}"
                    sandbox.write_file(file_patch.patched_content, container_path)
                    console.print(f"[green]✓ Applied to container: {file_patch.file_path}[/green]")
                elif repo_path:
                    # Write to local filesystem
                    full_path = Path(repo_path) / file_patch.file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(file_patch.patched_content, encoding='utf-8')
                    console.print(f"[green]✓ Applied locally: {file_patch.file_path}[/green]")
                else:
                    console.print(f"[red]✗ No target to write to for {file_patch.file_path}[/red]")
                    success = False
            except Exception as e:
                console.print(f"[red]✗ Failed to apply patch to {file_patch.file_path}: {e}[/red]")
                success = False

        return success

    def _read_file(self, file_path: str, repo_path: str, sandbox=None) -> Optional[str]:
        """Read file content from repo or sandbox."""
        try:
            if sandbox and sandbox.container:
                result = sandbox.run_command(f"cat /workspace/{file_path}")
                return result.stdout if result.success else None
            elif repo_path:
                full_path = Path(repo_path) / file_path
                if full_path.exists():
                    return full_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")
        return None

    def _compute_diff(self, original: str, patched: str, file_path: str) -> str:
        """Compute unified diff between original and patched content."""
        original_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        ))
        return '\n'.join(diff)

    def _format_plan(self, plan) -> str:
        """Format a RepairPlan into a readable string for the prompt."""
        if not plan:
            return "No plan available"
        try:
            return (
                f"Issue: {plan.issue_summary}\n"
                f"Root Cause: {plan.root_cause}\n"
                f"Strategy: {plan.selected_strategy.approach}\n"
                f"Target Functions: {', '.join(plan.target_functions)}\n"
                f"Expected Outcome: {plan.expected_test_outcome}\n"
                f"Risk Flags: {', '.join(plan.risk_flags) if plan.risk_flags else 'None'}"
            )
        except AttributeError:
            return str(plan)

    def _format_context(self, chunks: list[dict]) -> str:
        """Format context chunks for the prompt."""
        if not chunks:
            return "No additional context."
        parts = []
        for c in chunks:
            parts.append(
                f"### {c.get('name', 'unknown')} ({c.get('file_path', '')})\n"
                f"```python\n{c.get('source_code', '')[:300]}\n```"
            )
        return '\n\n'.join(parts)

    def _display_patch(self, patch: MultiFilePatch):
        """Display the patch in the terminal."""
        for file_patch in patch.patches:
            if file_patch.diff_text:
                console.print(f"\n[bold]Diff: {file_patch.file_path}[/bold]")
                syntax = Syntax(
                    file_patch.diff_text[:1500],
                    "diff",
                    theme="monokai",
                    line_numbers=False
                )
                console.print(syntax)
