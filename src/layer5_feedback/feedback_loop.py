"""
Layer 5: Test-Driven Feedback Loop
=====================================
Runs pytest inside the sandbox, parses structured JSON test reports,
extracts failure semantics, and feeds targeted retry prompts to Layer 4.

Executes up to 8 correction iterations with divergence detection.

Technologies: pytest, pytest-json-report, LangGraph conditional edges
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class TestFailure:
    """Structured representation of a single test failure."""
    test_name: str
    file_path: str
    line_number: int
    assertion_message: str
    expected_value: str
    actual_value: str
    traceback: str

    def to_prompt_text(self) -> str:
        """Format for LLM retry prompt."""
        return (
            f"FAILING TEST: {self.test_name}\n"
            f"Location: {self.file_path}:{self.line_number}\n"
            f"Assertion: {self.assertion_message}\n"
            f"Expected: {self.expected_value}\n"
            f"Actual: {self.actual_value}\n"
            f"Traceback:\n{self.traceback[:500]}"
        )


@dataclass
class TestRunResult:
    """Result from running the full test suite."""
    total_tests: int
    passed: int
    failed: int
    errors: int
    failures: list[TestFailure] = field(default_factory=list)
    raw_output: str = ""
    flaky_tests: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def summary(self) -> str:
        status = "✅ ALL PASS" if self.all_passed else f"❌ {self.failed} FAILING"
        return f"{status} | {self.passed}/{self.total_tests} tests passed"


class TestRunner:
    """
    Runs tests inside the Layer 3 sandbox and parses results.
    
    KEY: Uses pytest-json-report for machine-readable output,
    not human-readable terminal output.
    """

    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.flaky_run_count = 3  # Run failing tests N times to detect flakiness

    def run_tests(
        self,
        test_path: str = "tests/",
        working_dir: str = "/workspace"
    ) -> TestRunResult:
        """
        Run the full test suite inside the sandbox.
        
        Returns structured TestRunResult with failure details.
        """
        console.print("[cyan]🧪 Running test suite...[/cyan]")

        # Run pytest with JSON report for machine-readable output
        result = self.sandbox.run_command(
            f"cd {working_dir} && python -m pytest {test_path} "
            f"--json-report --json-report-file=/tmp/test_report.json "
            f"-x --tb=short -q 2>&1",
            workdir=working_dir
        )

        # Try to read the JSON report
        json_result = self.sandbox.run_command(
            "cat /tmp/test_report.json 2>/dev/null || echo '{}'",
            workdir=working_dir
        )

        test_run = self._parse_test_results(json_result.stdout, result.stdout)

        # Print summary table
        self._print_summary(test_run)
        return test_run

    def _parse_test_results(self, json_output: str, raw_output: str) -> TestRunResult:
        """Parse pytest-json-report output into structured TestRunResult."""
        try:
            data = json.loads(json_output.strip())
        except json.JSONDecodeError:
            # Fallback: parse from raw output
            return self._parse_raw_output(raw_output)

        summary = data.get('summary', {})
        failures = []

        for test in data.get('tests', []):
            if test.get('outcome') == 'failed':
                call_info = test.get('call', {})
                crash_info = call_info.get('crash', {})
                
                # Extract expected vs actual from assertion message
                longrepr = call_info.get('longrepr', '')
                expected, actual = self._extract_expected_actual(longrepr)

                failure = TestFailure(
                    test_name=test.get('nodeid', 'unknown'),
                    file_path=crash_info.get('path', 'unknown'),
                    line_number=crash_info.get('lineno', 0),
                    assertion_message=crash_info.get('message', longrepr[:200]),
                    expected_value=expected,
                    actual_value=actual,
                    traceback=longrepr[:1000]
                )
                failures.append(failure)

        return TestRunResult(
            total_tests=summary.get('total', 0),
            passed=summary.get('passed', 0),
            failed=summary.get('failed', 0),
            errors=summary.get('error', 0),
            failures=failures,
            raw_output=raw_output
        )

    def _parse_raw_output(self, raw: str) -> TestRunResult:
        """Fallback parser for when JSON report is unavailable."""
        lines = raw.split('\n')
        passed = failed = errors = 0

        for line in lines:
            if 'passed' in line and ('failed' in line or 'error' in line or line.strip().startswith('=')):
                # Parse lines like "5 passed, 2 failed"
                import re
                passed_match = re.search(r'(\d+) passed', line)
                failed_match = re.search(r'(\d+) failed', line)
                error_match = re.search(r'(\d+) error', line)
                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))

        return TestRunResult(
            total_tests=passed + failed + errors,
            passed=passed,
            failed=failed,
            errors=errors,
            raw_output=raw
        )

    def _extract_expected_actual(self, longrepr: str) -> tuple[str, str]:
        """Extract expected and actual values from pytest assertion message."""
        expected = actual = "unknown"
        lines = longrepr.split('\n')
        
        for line in lines:
            if 'assert' in line.lower() and '==' in line:
                parts = line.split('==')
                if len(parts) == 2:
                    actual = parts[0].replace('assert', '').strip()
                    expected = parts[1].strip()
            elif 'expected' in line.lower():
                expected = line.split(':', 1)[-1].strip()
            elif 'got' in line.lower() or 'actual' in line.lower():
                actual = line.split(':', 1)[-1].strip()

        return expected[:200], actual[:200]

    def detect_flaky_tests(self, failing_tests: list[str]) -> list[str]:
        """
        Run each failing test 3 times to identify flaky tests.
        A test is flaky if it passes in at least 1 of 3 runs.
        """
        flaky = []
        for test_name in failing_tests:
            passes = 0
            for _ in range(self.flaky_run_count):
                result = self.sandbox.run_command(
                    f"python -m pytest {test_name} -q 2>&1"
                )
                if result.success:
                    passes += 1
            
            # If it passes at least once, it's flaky
            if 0 < passes < self.flaky_run_count:
                flaky.append(test_name)
                console.print(f"[yellow]⚠ Flaky test detected: {test_name} "
                              f"({passes}/{self.flaky_run_count} runs passed)[/yellow]")
        
        return flaky

    def _print_summary(self, result: TestRunResult):
        """Print a rich summary table."""
        table = Table(title="Test Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Total Tests", str(result.total_tests))
        table.add_row("Passed", f"[green]{result.passed}[/green]")
        table.add_row("Failed", f"[red]{result.failed}[/red]" if result.failed else "0")
        table.add_row("Errors", f"[red]{result.errors}[/red]" if result.errors else "0")
        table.add_row("Status", result.summary())
        
        console.print(table)


class FeedbackLoop:
    """
    Layer 5: The retry loop that makes the agent genuinely autonomous.
    
    Orchestrates: test → analyze failures → generate retry prompt → 
                  send to Layer 4 → repeat up to MAX_ITERATIONS times.
    """

    MAX_ITERATIONS = 8
    DIVERGENCE_SIMILARITY_THRESHOLD = 0.95

    def __init__(self, sandbox, code_generator):
        """
        Args:
            sandbox: Layer 3 DockerSandbox instance
            code_generator: Layer 4 CodeGenerator instance
        """
        self.sandbox = sandbox
        self.code_generator = code_generator
        self.test_runner = TestRunner(sandbox)
        self.iteration_history: list[TestRunResult] = []

    def run(self, plan, repo_path: str) -> tuple[bool, int, list[TestRunResult]]:
        """
        Main feedback loop: run tests, retry if failing, stop when passing.
        
        Args:
            plan: RepairPlan from Layer 2
            repo_path: Path to repository (inside sandbox)
            
        Returns:
            (success: bool, iterations_used: int, history: list[TestRunResult])
        """
        console.print(f"\n[bold cyan]🔄 Starting feedback loop (max {self.MAX_ITERATIONS} iterations)[/bold cyan]")
        self.iteration_history = []

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            console.print(f"\n[bold]── Iteration {iteration}/{self.MAX_ITERATIONS} ──[/bold]")

            # Run tests
            test_result = self.test_runner.run_tests(working_dir=repo_path)
            self.iteration_history.append(test_result)

            # Check success
            if test_result.all_passed:
                console.print(f"[bold green]✅ All tests pass after {iteration} iteration(s)![/bold green]")
                return True, iteration, self.iteration_history

            # Check for divergence (stuck in a loop)
            if self._is_diverging():
                console.print("[yellow]⚠ Divergence detected — agent is stuck. Triggering backtrack.[/yellow]")
                return False, iteration, self.iteration_history

            # Check flaky tests (don't retry for flaky failures)
            flaky = self.test_runner.detect_flaky_tests(
                [f.test_name for f in test_result.failures]
            )
            genuine_failures = [f for f in test_result.failures
                               if f.test_name not in flaky]

            if not genuine_failures:
                console.print("[cyan]Only flaky tests remaining — treating as pass[/cyan]")
                return True, iteration, self.iteration_history

            # Generate retry prompt and get new patch from Layer 4
            retry_prompt = self._build_retry_prompt(genuine_failures, iteration)
            console.print(f"[cyan]Sending retry prompt to code generator ({len(genuine_failures)} failures)...[/cyan]")
            
            # Layer 4 generates a new patch based on the failure context
            new_patch = self.code_generator.retry_with_feedback(
                plan=plan,
                failures=genuine_failures,
                retry_prompt=retry_prompt,
                iteration=iteration
            )

            if new_patch:
                # Apply the new patch
                apply_result = self.code_generator.apply_patch(new_patch, sandbox=self.sandbox)
                if not apply_result:
                    console.print(f"[red]✗ Failed to apply patch on iteration {iteration}[/red]")

        console.print(f"[red]✗ Max iterations ({self.MAX_ITERATIONS}) reached without resolution[/red]")
        return False, self.MAX_ITERATIONS, self.iteration_history

    def _build_retry_prompt(self, failures: list[TestFailure], iteration: int) -> str:
        """Build a targeted retry prompt explaining what went wrong."""
        failure_texts = '\n\n'.join(f.to_prompt_text() for f in failures[:3])
        
        return f"""## Iteration {iteration} — Fix Required

Your previous patch caused {len(failures)} test(s) to fail.
Do NOT repeat the same approach — analyze the failures carefully and try a different fix.

## Failing Tests

{failure_texts}

## Instructions
1. Read the traceback carefully — understand WHY this specific assertion fails
2. Look at the line numbers to identify exactly where the problem is
3. Generate a NEW unified diff patch that fixes these failures WITHOUT breaking passing tests
4. Be minimal — change only what is necessary

Generate the corrected patch now:"""

    def _is_diverging(self) -> bool:
        """
        Detect if the agent is stuck (same error repeating).
        
        Checks if the last 3 failure messages are similar (>95% similarity).
        Simple version: checks if exact failure count and names are the same.
        """
        if len(self.iteration_history) < 3:
            return False

        # Compare last 3 iterations' failure names
        last_three = self.iteration_history[-3:]
        failure_sets = [
            frozenset(f.test_name for f in r.failures)
            for r in last_three
        ]

        # If all 3 have identical failing tests, we're diverging
        if len(set(failure_sets)) == 1 and all(len(s) > 0 for s in failure_sets):
            return True

        return False
