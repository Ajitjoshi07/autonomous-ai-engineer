"""
Demo: Full Pipeline End-to-End
================================
This script demonstrates the complete Autonomous AI Engineer pipeline
on a simple synthetic bug — no API key required for the basic demo!

Run: python scripts/demo_run.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


# ── Sample Buggy Code ────────────────────────────────────────────────────────

BUGGY_CODE = '''
def calculate_discount(price: float, discount_rate: float) -> float:
    """Calculate final price after applying discount rate."""
    if discount_rate:
        return price * discount_rate  # BUG: should be price * (1 - discount_rate)
    return 0  # BUG: should return price when discount_rate is 0


def apply_bulk_discount(prices: list, rate: float) -> list:
    """Apply discount to a list of prices."""
    return [calculate_discount(p, rate) for p in prices]
'''

TEST_CODE = '''
import pytest
from calculator import calculate_discount, apply_bulk_discount

def test_normal_discount():
    # 20% discount on $100 should give $80
    result = calculate_discount(100, 0.20)
    assert result == 80.0, f"Expected 80.0, got {result}"

def test_zero_discount():
    # 0% discount should return original price
    result = calculate_discount(100, 0)
    assert result == 100.0, f"Expected 100.0, got {result}"

def test_full_discount():
    # 100% discount should return 0
    result = calculate_discount(100, 1.0)
    assert result == 0.0, f"Expected 0.0, got {result}"

def test_bulk_discount():
    prices = [100, 200, 300]
    results = apply_bulk_discount(prices, 0.10)
    assert results == [90.0, 180.0, 270.0], f"Got {results}"
'''

FIXED_CODE = '''
def calculate_discount(price: float, discount_rate: float) -> float:
    """Calculate final price after applying discount rate."""
    if discount_rate >= 1.0:
        return 0.0
    return price * (1 - discount_rate)


def apply_bulk_discount(prices: list, rate: float) -> list:
    """Apply discount to a list of prices."""
    return [calculate_discount(p, rate) for p in prices]
'''

SAMPLE_ISSUE = """
**Bug Report: calculate_discount() returns wrong values**

**Environment:** Python 3.11, calculator.py

**Expected Behavior:**
- `calculate_discount(100, 0.20)` → `80.0` (20% off = pay 80%)  
- `calculate_discount(100, 0)` → `100.0` (0% off = pay full price)
- `calculate_discount(100, 1.0)` → `0.0` (100% off = free)

**Actual Behavior:**
- `calculate_discount(100, 0.20)` → `20.0` ❌ (returning discount amount, not discounted price)
- `calculate_discount(100, 0)` → `0` ❌ (should return 100, not 0)

**Steps to Reproduce:**
```python
from calculator import calculate_discount
print(calculate_discount(100, 0.20))  # prints 20.0 instead of 80.0
print(calculate_discount(100, 0))     # prints 0 instead of 100
```

**Priority:** High — this affects all pricing calculations in production.
"""


def demo_layer1():
    """Demo: Codebase Understanding Engine."""
    console.print(Rule("[bold cyan]Layer 1: Codebase Understanding[/bold cyan]"))
    console.print("Parsing code with tree-sitter...")
    console.print("Embedding functions with CodeBERT...")
    console.print("Building call graph with NetworkX...")
    
    # Show what would be retrieved
    console.print(Panel(
        "[green]Retrieved chunks for: 'function that calculates discount'[/green]\n\n"
        "1. [yellow]calculate_discount[/yellow] (calculator.py:1) — score: 0.94\n"
        "2. [yellow]apply_bulk_discount[/yellow] (calculator.py:8) — score: 0.81\n"
        "3. [yellow]test_normal_discount[/yellow] (test_calculator.py:4) — score: 0.72",
        title="FAISS Retrieval Results",
        border_style="cyan"
    ))


def demo_layer2():
    """Demo: Tree-of-Thought Planning."""
    console.print(Rule("[bold cyan]Layer 2: Tree-of-Thought Planning[/bold cyan]"))
    
    console.print(Panel(
        "[bold]Strategy A[/bold] (confidence: 0.89)\n"
        "Hypothesis: calculate_discount() returns discount amount instead of discounted price\n"
        "Fix: Change return to `price * (1 - discount_rate)`\n"
        "Risk: LOW\n\n"
        "[bold]Strategy B[/bold] (confidence: 0.61)\n"
        "Hypothesis: discount_rate parameter is being misused by callers\n"
        "Fix: Add input validation and docstring clarification\n"
        "Risk: MEDIUM\n\n"
        "[bold]Strategy C[/bold] (confidence: 0.34)\n"
        "Hypothesis: The zero-case is handled separately by caller\n"
        "Fix: Only fix the zero return value\n"
        "Risk: HIGH\n\n"
        "→ [green]Selected: Strategy A (highest confidence)[/green]",
        title="Tree-of-Thought Strategies",
        border_style="cyan"
    ))


def demo_layer3():
    """Demo: Docker Sandbox."""
    console.print(Rule("[bold cyan]Layer 3: Secure Sandbox[/bold]"))
    console.print(Panel(
        "🐳 Container spawned: python:3.11-slim\n"
        "   --network none ✅\n"
        "   --memory 512m ✅\n"
        "   --cpus 0.5 ✅\n"
        "   tmpfs /workspace ✅\n"
        "   Watchdog: 30s timeout ✅\n\n"
        "Repository copied to /workspace",
        title="Sandbox Configuration",
        border_style="cyan"
    ))


def demo_layer4_and_5(tmp_dir: str):
    """Demo: Code Generation + Test Feedback Loop (ACTUALLY RUNS TESTS)."""
    import subprocess

    console.print(Rule("[bold cyan]Layer 4 + 5: Code Generation & Feedback Loop[/bold]"))

    # Write the buggy code
    calc_file = os.path.join(tmp_dir, 'calculator.py')
    test_file = os.path.join(tmp_dir, 'test_calculator.py')
    
    with open(calc_file, 'w') as f:
        f.write(BUGGY_CODE)
    with open(test_file, 'w') as f:
        f.write(TEST_CODE)

    # ITERATION 1: Run tests with buggy code
    console.print("\n[bold]── Iteration 1 ──[/bold]")
    console.print("Running tests with original (buggy) code...")
    
    result = subprocess.run(
        ['python', '-m', 'pytest', 'test_calculator.py', '-v', '--tb=short'],
        cwd=tmp_dir,
        capture_output=True,
        text=True
    )
    
    # Show results
    output_lines = [l for l in result.stdout.split('\n') if l.strip()]
    for line in output_lines[-15:]:
        if 'FAILED' in line:
            console.print(f"[red]{line}[/red]")
        elif 'PASSED' in line:
            console.print(f"[green]{line}[/green]")
        elif 'passed' in line or 'failed' in line:
            console.print(f"[bold]{line}[/bold]")
        else:
            console.print(line)

    console.print(Panel(
        "Tests failing!\n"
        "Layer 5 extracts failure semantics:\n\n"
        "• test_normal_discount: Expected 80.0, got 20.0\n"
        "• test_zero_discount: Expected 100.0, got 0\n"
        "• test_bulk_discount: Wrong values throughout\n\n"
        "→ Sending structured error context to Layer 4 for retry...",
        title="Failure Analysis",
        border_style="yellow"
    ))

    # ITERATION 2: Apply the fix
    console.print("\n[bold]── Iteration 2 (after LLM retry) ──[/bold]")
    console.print("Layer 4 generates new patch based on failure context...")
    console.print(Panel(
        "--- calculator.py\n"
        "+++ calculator.py\n"
        "@@ -2,4 +2,4 @@\n"
        ' def calculate_discount(price: float, discount_rate: float) -> float:\n'
        '-    if discount_rate:\n'
        '-        return price * discount_rate\n'
        '-    return 0\n'
        '+    if discount_rate >= 1.0:\n'
        '+        return 0.0\n'
        '+    return price * (1 - discount_rate)',
        title="Generated Patch (Unified Diff)",
        border_style="green"
    ))

    # Write the fixed code
    with open(calc_file, 'w') as f:
        f.write(FIXED_CODE)

    # Run tests with fixed code
    console.print("Running tests with fixed code...")
    result = subprocess.run(
        ['python', '-m', 'pytest', 'test_calculator.py', '-v'],
        cwd=tmp_dir,
        capture_output=True,
        text=True
    )
    
    output_lines = [l for l in result.stdout.split('\n') if l.strip()]
    for line in output_lines[-10:]:
        if 'FAILED' in line:
            console.print(f"[red]{line}[/red]")
        elif 'PASSED' in line:
            console.print(f"[green]{line}[/green]")
        elif 'passed' in line:
            console.print(f"[bold green]{line}[/bold green]")
        else:
            console.print(line)

    return result.returncode == 0


def demo_layer6():
    """Demo: Self-Critique Agent."""
    console.print(Rule("[bold cyan]Layer 6: Self-Critique & Quality[/bold]"))
    console.print(Panel(
        "Critic LLM reviewing patch...\n\n"
        "✅ Correctness: Fix correctly handles zero-rate edge case\n"
        "✅ Regression risk: Change is isolated to calculate_discount()\n"
        "✅ Code style: Matches project conventions\n"
        "✅ Simplicity: Clean one-liner solution\n"
        "✅ Security: No new vulnerabilities introduced\n\n"
        "ruff linter: 0 violations\n\n"
        "→ [green]APPROVED — proceeding to PR creation[/green]",
        title="Critic Review",
        border_style="cyan"
    ))


def demo_layer7():
    """Demo: Memory & Self-Improvement."""
    console.print(Rule("[bold cyan]Layer 7: Memory & Self-Improvement[/bold]"))
    console.print(Panel(
        "Storing session to episodic memory...\n\n"
        "Memory record:\n"
        "  issue_embedding: [768-dim vector]\n"
        "  root_cause: 'incorrect arithmetic in discount calculation'\n"
        "  fix_strategy: 'fix conditional + arithmetic formula'\n"
        "  iterations_required: 2\n"
        "  critic_approved: true\n\n"
        "Next time a similar arithmetic bug appears:\n"
        "→ Planner will prefer Strategy A from iteration 1\n"
        "→ Estimated savings: ~1 retry iteration",
        title="Episodic Memory Storage",
        border_style="cyan"
    ))


def main():
    console.print(Panel.fit(
        "[bold white]AUTONOMOUS AI SOFTWARE ENGINEER[/bold white]\n"
        "[dim]by Ajit Mukund Joshi — B.Tech AI & Data Science[/dim]\n\n"
        "End-to-End Pipeline Demo",
        border_style="bright_blue"
    ))

    console.print(Panel(
        SAMPLE_ISSUE.strip(),
        title="📋 Input: GitHub Issue",
        border_style="yellow"
    ))

    console.print("\n[bold]Starting pipeline...[/bold]\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_layer1()
        console.print()
        demo_layer2()
        console.print()
        demo_layer3()
        console.print()
        
        success = demo_layer4_and_5(tmp_dir)
        console.print()
        
        if success:
            demo_layer6()
            console.print()
            demo_layer7()
            console.print()

    # Final summary
    console.print(Panel(
        "[bold green]✅ Task Complete![/bold green]\n\n"
        "• Branch created:  agent-fix/issue-42-calculate-discount\n"
        "• Patch applied:   calculator.py (4 lines changed)\n"
        "• Tests passing:   4/4 ✅\n"
        "• Critic approved: Yes\n"
        "• Iterations used: 2\n"
        "• Time elapsed:    ~4 minutes\n"
        "• API cost:        ~$0.08\n\n"
        "Pull Request would be opened at:\n"
        "github.com/repo/pull/43",
        title="🚀 Pull Request Ready",
        border_style="green"
    ))

    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("1. Get a FREE Groq API key: https://console.groq.com")
    console.print("2. Create GitHub Personal Access Token: github.com/settings/tokens")
    console.print("3. Fill in config/.env")
    console.print("4. Run: python src/layer1_understanding/engine.py /path/to/repo")
    console.print("5. Install Docker Desktop: docker.com/products/docker-desktop")


if __name__ == "__main__":
    main()
