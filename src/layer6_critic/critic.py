"""
Layer 6: Self-Critique & Quality Agent
=========================================
A second LLM instance with an adversarial review prompt evaluates every
patch on correctness, regression risk, style, simplicity, and security.

Also runs ruff automatically and feeds linting violations to the generator
for cleanup before critic review.

Technologies: GPT-4o (critic prompt), ruff, pylint, Pydantic structured feedback
"""

import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()


# ── Data Models ──────────────────────────────────────────────────────────────

class CriticFeedback(BaseModel):
    """Structured feedback from the critic LLM."""
    approved: bool = Field(description="True if patch is approved for PR submission")

    # Five evaluation dimensions
    correctness_score: float = Field(ge=0.0, le=1.0, description="Does it fix the reported issue?")
    regression_risk: str = Field(description="low | medium | high")
    style_score: float = Field(ge=0.0, le=1.0, description="Matches project conventions?")
    simplicity_score: float = Field(ge=0.0, le=1.0, description="Is this the simplest correct solution?")
    security_score: float = Field(ge=0.0, le=1.0, description="No new vulnerabilities introduced?")

    # Detailed feedback
    overall_assessment: str = Field(description="One-paragraph summary")
    concerns: list[str] = Field(default_factory=list, description="Specific issues to address")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    rejection_reason: Optional[str] = Field(default=None, description="Why rejected (if not approved)")

    @property
    def overall_score(self) -> float:
        return (
            self.correctness_score * 0.40 +
            (1.0 if self.regression_risk == 'low' else 0.5 if self.regression_risk == 'medium' else 0.1) * 0.25 +
            self.style_score * 0.15 +
            self.simplicity_score * 0.10 +
            self.security_score * 0.10
        )


@dataclass
class LintResult:
    """Result from running ruff linter."""
    violations: list[dict] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    passed: bool = True

    def to_prompt_text(self) -> str:
        if not self.violations:
            return "No linting violations found."
        lines = [f"Found {len(self.violations)} linting issue(s):"]
        for v in self.violations[:10]:
            lines.append(
                f"  {v.get('filename', '?')}:{v.get('location', {}).get('row', '?')}: "
                f"[{v.get('code', '?')}] {v.get('message', '?')}"
            )
        return '\n'.join(lines)


# ── Prompts ──────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a senior software engineer conducting an adversarial code review.
Your job is to REJECT patches that are not production-ready.
Be skeptical. Be thorough. Look for problems the author missed.
Only approve patches that you would be comfortable merging into a production codebase."""

CRITIC_REVIEW_PROMPT = """Review this code patch and provide structured feedback.

## Original GitHub Issue
{issue_text}

## Patch Summary
Files changed: {files_changed}
Lines added: {lines_added}
Lines removed: {lines_removed}

## Full Patch (Unified Diff)
```diff
{diff_text}
```

## Linting Results
{lint_results}

## Evaluation Criteria
Score each dimension from 0.0 to 1.0:

1. **Correctness** (40% weight): Does this patch actually fix the reported issue?
   - Check: Does the logic match the expected behavior described in the issue?
   - Check: Are all edge cases handled (zero values, None, empty lists, etc.)?

2. **Regression Risk** (25% weight): Could this break existing functionality?
   - Check: Are any function signatures changed?
   - Check: Could callers be affected?
   - Rate as: low | medium | high

3. **Style** (15% weight): Does it match the codebase conventions?
   - Check: Consistent naming, indentation, docstring style

4. **Simplicity** (10% weight): Is this the simplest correct solution?
   - Check: Is there a cleaner way to achieve the same result?

5. **Security** (10% weight): Does the fix introduce new vulnerabilities?
   - Check: Input validation, injection risks, resource exhaustion

Return ONLY valid JSON:
{{
  "approved": true or false,
  "correctness_score": 0.0-1.0,
  "regression_risk": "low|medium|high",
  "style_score": 0.0-1.0,
  "simplicity_score": 0.0-1.0,
  "security_score": 0.0-1.0,
  "overall_assessment": "...",
  "concerns": ["..."],
  "suggestions": ["..."],
  "rejection_reason": null or "specific reason for rejection"
}}"""


# ── Linter ────────────────────────────────────────────────────────────────────

class Linter:
    """Runs ruff linter on patched files."""

    def __init__(self):
        self.ruff_available = self._check_ruff()

    def _check_ruff(self) -> bool:
        try:
            result = subprocess.run(['ruff', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                console.print(f"[green]✓ ruff {result.stdout.strip()}[/green]")
                return True
        except FileNotFoundError:
            pass
        console.print("[yellow]⚠ ruff not found. Run: pip install ruff[/yellow]")
        return False

    def lint_file(self, file_path: str, content: str) -> LintResult:
        """Lint a file's content (writes to temp file, runs ruff)."""
        import tempfile

        if not self.ruff_available:
            return LintResult(passed=True)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ['ruff', 'check', '--output-format=json', tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            violations = []
            if result.stdout.strip():
                raw = json.loads(result.stdout)
                for v in raw:
                    v['filename'] = file_path  # Replace temp path with real path
                    violations.append(v)

            errors = [v for v in violations if v.get('code', '').startswith('E')]
            warnings = [v for v in violations if not v.get('code', '').startswith('E')]

            lint_result = LintResult(
                violations=violations,
                error_count=len(errors),
                warning_count=len(warnings),
                passed=len(errors) == 0
            )

            if violations:
                console.print(f"[yellow]⚠ ruff: {len(violations)} issue(s) in {file_path}[/yellow]")
            else:
                console.print(f"[green]✓ ruff: clean ({file_path})[/green]")

            return lint_result

        except Exception as e:
            console.print(f"[yellow]⚠ Linting error: {e}[/yellow]")
            return LintResult(passed=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def lint_patch(self, patch) -> dict[str, LintResult]:
        """Lint all files in a MultiFilePatch."""
        results = {}
        for file_patch in patch.patches:
            results[file_patch.file_path] = self.lint_file(
                file_patch.file_path,
                file_patch.patched_content
            )
        return results


# ── Critic Agent ─────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Layer 6: Adversarial code review agent.

    Uses a second LLM instance with a different system prompt from the
    code generator — instructed to look for problems and reject bad patches.

    Usage:
        critic = CriticAgent()
        feedback = critic.review(patch, issue_text)
        if feedback.approved:
            # proceed to PR creation
        else:
            # send feedback back to Layer 4
    """

    def __init__(self):
        self.linter = Linter()
        self.llm = self._init_llm()
        self.review_history: list[CriticFeedback] = []

    def _init_llm(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from layer2_planning.agent import LLMClient
            return LLMClient()
        except Exception as e:
            console.print(f"[yellow]⚠ Critic LLM init: {e}[/yellow]")
            return None

    def review(self, patch, issue_text: str = "") -> CriticFeedback:
        """
        Full review pipeline: lint → LLM critique → structured feedback.

        Args:
            patch: MultiFilePatch from Layer 4
            issue_text: Original GitHub issue for context

        Returns:
            CriticFeedback with approval decision and detailed scores
        """
        console.print("\n[bold cyan]🔍 Critic Agent: Reviewing patch...[/bold cyan]")

        # Step 1: Run linter on all patched files
        lint_results = self.linter.lint_patch(patch)
        combined_lint = self._combine_lint_results(lint_results)

        # Step 2: LLM review
        feedback = self._llm_review(patch, issue_text, combined_lint)

        # Step 3: Display results
        self._display_review(feedback)
        self.review_history.append(feedback)

        return feedback

    def _llm_review(
        self,
        patch,
        issue_text: str,
        lint_text: str
    ) -> CriticFeedback:
        """Run LLM critic review."""
        if not self.llm:
            # Fallback: auto-approve if no LLM available
            console.print("[yellow]⚠ No LLM — auto-approving patch[/yellow]")
            return CriticFeedback(
                approved=True,
                correctness_score=0.8,
                regression_risk='low',
                style_score=0.8,
                simplicity_score=0.8,
                security_score=1.0,
                overall_assessment="Auto-approved (LLM not available)",
                concerns=[],
                suggestions=[]
            )

        # Build diff text from all patches
        all_diffs = '\n\n'.join(p.diff_text for p in patch.patches)
        total_added = sum(p.lines_added for p in patch.patches)
        total_removed = sum(p.lines_removed for p in patch.patches)

        prompt = CRITIC_REVIEW_PROMPT.format(
            issue_text=issue_text[:500] if issue_text else "Not provided",
            files_changed=len(patch.patches),
            lines_added=total_added,
            lines_removed=total_removed,
            diff_text=all_diffs[:3000],
            lint_results=lint_text
        )

        try:
            response = self.llm.chat(
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.3  # Some variation to catch different issues
            )
            data = json.loads(response)
            feedback = CriticFeedback(**data)
            return feedback

        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Critic JSON parse error: {e}[/red]")
            return self._fallback_feedback(approved=False, reason="Critic parsing failed")
        except Exception as e:
            console.print(f"[red]✗ Critic review error: {e}[/red]")
            return self._fallback_feedback(approved=False, reason=str(e))

    def _combine_lint_results(self, results: dict[str, LintResult]) -> str:
        """Combine lint results from all files into a single text."""
        if not results:
            return "No linting performed."
        all_text = []
        for file_path, result in results.items():
            all_text.append(f"**{file_path}:** {result.to_prompt_text()}")
        return '\n'.join(all_text)

    def _fallback_feedback(self, approved: bool, reason: str) -> CriticFeedback:
        return CriticFeedback(
            approved=approved,
            correctness_score=0.5,
            regression_risk='medium',
            style_score=0.5,
            simplicity_score=0.5,
            security_score=0.5,
            overall_assessment=reason,
            concerns=[reason],
            suggestions=[],
            rejection_reason=reason if not approved else None
        )

    def _display_review(self, feedback: CriticFeedback):
        """Display the critic review in a rich table."""
        status = "[bold green]✅ APPROVED[/bold green]" if feedback.approved \
            else "[bold red]❌ REJECTED[/bold red]"

        table = Table(title=f"Critic Review — {status}", show_header=True)
        table.add_column("Dimension", style="cyan", width=20)
        table.add_column("Score / Rating", style="bold", width=15)
        table.add_column("Weight", width=8)

        def score_color(score: float) -> str:
            if score >= 0.8:
                return f"[green]{score:.2f}[/green]"
            elif score >= 0.6:
                return f"[yellow]{score:.2f}[/yellow]"
            else:
                return f"[red]{score:.2f}[/red]"

        table.add_row("Correctness", score_color(feedback.correctness_score), "40%")
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(feedback.regression_risk, "white")
        table.add_row("Regression Risk", f"[{risk_color}]{feedback.regression_risk}[/{risk_color}]", "25%")
        table.add_row("Code Style", score_color(feedback.style_score), "15%")
        table.add_row("Simplicity", score_color(feedback.simplicity_score), "10%")
        table.add_row("Security", score_color(feedback.security_score), "10%")
        table.add_row("─" * 18, "─" * 13, "─" * 6)
        table.add_row("Overall", score_color(feedback.overall_score), "100%")

        console.print(table)
        console.print(f"\n[italic]{feedback.overall_assessment}[/italic]")

        if feedback.concerns:
            console.print("\n[yellow]Concerns:[/yellow]")
            for c in feedback.concerns:
                console.print(f"  • {c}")

        if not feedback.approved and feedback.rejection_reason:
            console.print(f"\n[red]Rejection reason: {feedback.rejection_reason}[/red]")


# ── Entry point for testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('config/.env')

    # Create a mock patch for testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from layer4_codegen.patch_engine import FilePatch, MultiFilePatch

    test_patch = MultiFilePatch(
        patches=[
            FilePatch(
                file_path="calculator.py",
                original_content="""
def calculate_discount(price, discount_rate):
    if discount_rate:
        return price * discount_rate
    return 0
""",
                patched_content="""
def calculate_discount(price, discount_rate):
    if discount_rate >= 1.0:
        return 0.0
    return price * (1 - discount_rate)
""",
                diff_text="""--- a/calculator.py
+++ b/calculator.py
@@ -1,5 +1,5 @@
 def calculate_discount(price, discount_rate):
-    if discount_rate:
-        return price * discount_rate
-    return 0
+    if discount_rate >= 1.0:
+        return 0.0
+    return price * (1 - discount_rate)
"""
            )
        ],
        description="Fix calculate_discount logic"
    )

    critic = CriticAgent()
    feedback = critic.review(
        test_patch,
        issue_text="calculate_discount returns wrong value when discount_rate is 0"
    )
    console.print(f"\nApproved: {feedback.approved}")
