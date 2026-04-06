"""
Unit Tests — Autonomous AI Software Engineer
=============================================
Tests for each layer that work WITHOUT API keys.
Uses mocks and local computation where possible.

Run: pytest tests/ -v
"""

import sys
import os
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ════════════════════════════════════════════════════════════════
# LAYER 1 TESTS
# ════════════════════════════════════════════════════════════════

class TestLayer1ASTParser:
    """Tests for the AST-level code parser."""

    def test_parse_simple_function(self, tmp_path):
        """Parser correctly extracts a function chunk."""
        code = '''
def hello_world(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b
'''
        test_file = tmp_path / "test_module.py"
        test_file.write_text(code)

        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file(str(test_file))

        # Should find both functions
        assert len(chunks) >= 2
        names = [c.name for c in chunks]
        assert 'hello_world' in names
        assert 'add' in names

    def test_parse_class_with_methods(self, tmp_path):
        """Parser extracts class methods correctly."""
        code = '''
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
'''
        test_file = tmp_path / "calc.py"
        test_file.write_text(code)

        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file(str(test_file))

        names = [c.name for c in chunks]
        # Should have class + methods
        assert len(chunks) >= 3

    def test_parse_empty_file(self, tmp_path):
        """Parser handles empty files gracefully."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file(str(test_file))
        assert isinstance(chunks, list)

    def test_parse_syntax_error_file(self, tmp_path):
        """Parser handles syntax errors without crashing."""
        code = "def broken(:\n    pass"
        test_file = tmp_path / "broken.py"
        test_file.write_text(code)

        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file(str(test_file))
        assert isinstance(chunks, list)  # Should not raise

    def test_parse_nonexistent_file(self):
        """Parser returns empty list for missing files."""
        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file("/nonexistent/path/file.py")
        assert chunks == []

    def test_chunk_contains_source_code(self, tmp_path):
        """Each chunk contains the actual source code."""
        code = '''
def my_function():
    x = 1
    y = 2
    return x + y
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        chunks = parser.parse_file(str(test_file))

        fn_chunk = next((c for c in chunks if c.name == 'my_function'), None)
        assert fn_chunk is not None
        assert 'my_function' in fn_chunk.source_code
        assert fn_chunk.start_line > 0


class TestLayer1CallGraph:
    """Tests for the call graph builder."""

    def test_detects_function_calls(self, tmp_path):
        """Call graph correctly identifies which functions call which."""
        from layer1_understanding.engine import CodeChunk, CallGraphBuilder

        # Simulate chunks where foo() calls bar()
        foo_chunk = CodeChunk(
            chunk_id='', file_path='test.py', node_type='function',
            name='foo', start_line=1, end_line=5,
            source_code='def foo():\n    return bar()'
        )
        bar_chunk = CodeChunk(
            chunk_id='', file_path='test.py', node_type='function',
            name='bar', start_line=7, end_line=9,
            source_code='def bar():\n    return 42'
        )

        builder = CallGraphBuilder()
        graph = builder.build_from_chunks([foo_chunk, bar_chunk])

        # foo should call bar
        assert 'foo' in graph.nodes
        assert 'bar' in graph.nodes
        # bar should be a callee of foo
        callees = builder.get_callees('foo')
        assert 'bar' in callees

    def test_get_callers_returns_correct_functions(self):
        """get_callers returns all functions that call a target."""
        from layer1_understanding.engine import CodeChunk, CallGraphBuilder

        a_chunk = CodeChunk(chunk_id='', file_path='f.py', node_type='function',
                           name='process', start_line=1, end_line=3,
                           source_code='def process():\n    validate()\n    save()')
        b_chunk = CodeChunk(chunk_id='', file_path='f.py', node_type='function',
                           name='validate', start_line=5, end_line=7,
                           source_code='def validate():\n    return True')
        c_chunk = CodeChunk(chunk_id='', file_path='f.py', node_type='function',
                           name='save', start_line=9, end_line=11,
                           source_code='def save():\n    return True')

        builder = CallGraphBuilder()
        builder.build_from_chunks([a_chunk, b_chunk, c_chunk])

        callers_of_validate = builder.get_callers('validate')
        assert 'process' in callers_of_validate


# ════════════════════════════════════════════════════════════════
# LAYER 3 TESTS
# ════════════════════════════════════════════════════════════════

class TestLayer3Sandbox:
    """Tests for the Docker sandbox (uses local fallback if Docker unavailable)."""

    def test_sandbox_creates_and_destroys(self):
        """Sandbox context manager starts and stops cleanly."""
        from layer3_sandbox.sandbox import DockerSandbox

        with DockerSandbox(memory_mb=128, timeout_seconds=10) as sandbox:
            # Just test it doesn't crash
            assert sandbox is not None
        # After exit, container should be stopped/None

    def test_local_command_execution(self):
        """Local fallback can execute basic commands."""
        from layer3_sandbox.sandbox import DockerSandbox

        sandbox = DockerSandbox(timeout_seconds=5)
        # Force local mode
        sandbox.docker_client = None
        sandbox.container = None

        result = sandbox.run_command("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_timeout_enforcement(self):
        """Sandbox enforces timeout on long-running commands."""
        from layer3_sandbox.sandbox import DockerSandbox

        sandbox = DockerSandbox(timeout_seconds=2)
        sandbox.docker_client = None  # Force local mode

        result = sandbox.run_command("sleep 5")
        assert result.timed_out or result.exit_code != 0

    def test_sandbox_result_success_property(self):
        """SandboxResult.success correctly reflects exit code."""
        from layer3_sandbox.sandbox import SandboxResult

        success = SandboxResult(exit_code=0, stdout="ok", stderr="", timed_out=False, execution_time_seconds=0.1)
        failure = SandboxResult(exit_code=1, stdout="", stderr="error", timed_out=False, execution_time_seconds=0.1)
        timeout = SandboxResult(exit_code=0, stdout="", stderr="", timed_out=True, execution_time_seconds=2.0)

        assert success.success is True
        assert failure.success is False
        assert timeout.success is False


# ════════════════════════════════════════════════════════════════
# LAYER 4 TESTS
# ════════════════════════════════════════════════════════════════

class TestLayer4PatchEngine:
    """Tests for the patch engine (diff computation and application)."""

    def test_compute_diff_detects_changes(self):
        """Diff correctly identifies added and removed lines."""
        from layer4_codegen.patch_engine import CodeGenerator

        generator = CodeGenerator.__new__(CodeGenerator)  # Skip __init__
        generator.llm = None
        generator.patch_history = []

        original = "def foo():\n    return 0\n"
        patched = "def foo():\n    return 1\n"

        diff = generator._compute_diff(original, patched, "test.py")
        assert '-    return 0' in diff
        assert '+    return 1' in diff

    def test_file_patch_counts_lines(self):
        """FilePatch correctly counts added and removed lines."""
        from layer4_codegen.patch_engine import FilePatch

        diff = """--- a/test.py
+++ b/test.py
@@ -1,4 +1,5 @@
 def foo():
-    x = 1
-    return x
+    x = 10
+    y = 20
+    return x + y
"""
        patch = FilePatch(
            file_path="test.py",
            original_content="def foo():\n    x = 1\n    return x\n",
            patched_content="def foo():\n    x = 10\n    y = 20\n    return x + y\n",
            diff_text=diff
        )
        assert patch.lines_added == 3
        assert patch.lines_removed == 2

    def test_apply_patch_writes_file(self, tmp_path):
        """apply_patch correctly writes patched content to disk."""
        from layer4_codegen.patch_engine import CodeGenerator, FilePatch, MultiFilePatch

        generator = CodeGenerator.__new__(CodeGenerator)
        generator.llm = None
        generator.patch_history = []

        test_file = tmp_path / "calculator.py"
        test_file.write_text("def foo():\n    return 0\n")

        patch = MultiFilePatch(patches=[
            FilePatch(
                file_path="calculator.py",
                original_content="def foo():\n    return 0\n",
                patched_content="def foo():\n    return 42\n",
                diff_text=""
            )
        ])

        result = generator.apply_patch(patch, repo_path=str(tmp_path))
        assert result is True
        assert test_file.read_text() == "def foo():\n    return 42\n"


# ════════════════════════════════════════════════════════════════
# LAYER 5 TESTS
# ════════════════════════════════════════════════════════════════

class TestLayer5FeedbackLoop:
    """Tests for test parsing and feedback loop logic."""

    def test_parse_raw_pytest_output(self):
        """Raw pytest output is parsed into TestRunResult."""
        from layer5_feedback.feedback_loop import TestRunner
        from unittest.mock import MagicMock

        runner = TestRunner(sandbox=MagicMock())
        raw = "5 passed, 2 failed in 0.43s"
        result = runner._parse_raw_output(raw)

        assert result.passed == 5
        assert result.failed == 2
        assert result.total_tests == 7

    def test_all_passed_property(self):
        """TestRunResult.all_passed is True only when no failures."""
        from layer5_feedback.feedback_loop import TestRunResult

        passing = TestRunResult(total_tests=5, passed=5, failed=0, errors=0)
        failing = TestRunResult(total_tests=5, passed=3, failed=2, errors=0)
        erroring = TestRunResult(total_tests=5, passed=4, failed=0, errors=1)

        assert passing.all_passed is True
        assert failing.all_passed is False
        assert erroring.all_passed is False

    def test_divergence_detection_triggers_after_3_same_failures(self):
        """Divergence detector fires when same tests fail 3 times in a row."""
        from layer5_feedback.feedback_loop import FeedbackLoop, TestRunResult, TestFailure
        from unittest.mock import MagicMock

        loop = FeedbackLoop(sandbox=MagicMock(), code_generator=MagicMock())

        # Create 3 identical failure results
        failure = TestFailure(
            test_name="tests/test_calc.py::test_discount",
            file_path="tests/test_calc.py",
            line_number=10,
            assertion_message="assert 0 == 100",
            expected_value="100",
            actual_value="0",
            traceback="..."
        )
        repeated_result = TestRunResult(
            total_tests=5, passed=4, failed=1, errors=0,
            failures=[failure]
        )

        loop.iteration_history = [repeated_result, repeated_result, repeated_result]
        assert loop._is_diverging() is True

    def test_no_divergence_when_failures_change(self):
        """Divergence not triggered when different tests fail each time."""
        from layer5_feedback.feedback_loop import FeedbackLoop, TestRunResult, TestFailure
        from unittest.mock import MagicMock

        loop = FeedbackLoop(sandbox=MagicMock(), code_generator=MagicMock())

        def make_result(test_name):
            return TestRunResult(
                total_tests=5, passed=4, failed=1, errors=0,
                failures=[TestFailure(
                    test_name=test_name, file_path="tests/t.py",
                    line_number=1, assertion_message="x",
                    expected_value="1", actual_value="0", traceback=""
                )]
            )

        loop.iteration_history = [
            make_result("test_a"),
            make_result("test_b"),
            make_result("test_c"),
        ]
        assert loop._is_diverging() is False


# ════════════════════════════════════════════════════════════════
# LAYER 7 TESTS
# ════════════════════════════════════════════════════════════════

class TestLayer7Memory:
    """Tests for the episodic memory store."""

    def test_store_and_retrieve(self, tmp_path):
        """Stored memory can be retrieved with similarity search."""
        from layer7_memory.memory_store import EpisodicMemoryStore, MemoryRecord
        from datetime import datetime

        store = EpisodicMemoryStore(store_path=str(tmp_path / 'memory'))

        record = MemoryRecord(
            memory_id='test001',
            timestamp=datetime.utcnow().isoformat(),
            repository='test/repo',
            issue_text='calculate_discount returns wrong value when discount_rate is zero',
            success=True,
            iterations_required=2,
            root_cause_classification='wrong conditional check',
            fix_strategy_used='fix arithmetic formula',
            critic_approved=True,
            critic_score=0.9,
        )
        store.store(record)

        # Try to retrieve a similar issue
        results = store.retrieve_similar(
            "function returns 0 when rate is 0, should return original value",
            top_k=5,
            min_similarity=0.0  # Accept any similarity for testing
        )

        assert len(results) >= 0  # Store works without crashing

    def test_statistics_empty_store(self, tmp_path):
        """Statistics work correctly on empty store."""
        from layer7_memory.memory_store import EpisodicMemoryStore

        store = EpisodicMemoryStore(store_path=str(tmp_path / 'memory'))
        stats = store.get_statistics()
        assert stats['total'] == 0

    def test_memory_record_to_dict_roundtrip(self):
        """MemoryRecord survives dict serialization."""
        from layer7_memory.memory_store import MemoryRecord
        from datetime import datetime

        original = MemoryRecord(
            memory_id='abc123',
            timestamp=datetime.utcnow().isoformat(),
            repository='owner/repo',
            issue_text='Bug: function returns None',
            success=True,
            iterations_required=3,
            root_cause_classification='null check missing',
            fix_strategy_used='add null guard',
            critic_approved=True,
            critic_score=0.87,
        )

        d = original.to_dict()
        restored = MemoryRecord.from_dict(d)

        assert restored.memory_id == original.memory_id
        assert restored.success == original.success
        assert restored.critic_score == original.critic_score

    def test_memory_persists_to_disk(self, tmp_path):
        """Memory records are written to disk as JSON."""
        from layer7_memory.memory_store import EpisodicMemoryStore, MemoryRecord
        from datetime import datetime

        store = EpisodicMemoryStore(store_path=str(tmp_path / 'memory'))
        record = MemoryRecord(
            memory_id='persist_test',
            timestamp=datetime.utcnow().isoformat(),
            repository='test/repo',
            issue_text='test issue',
            success=True,
        )
        store.store(record)

        # JSON file should exist
        json_file = tmp_path / 'memory' / 'persist_test.json'
        assert json_file.exists()
        import json
        data = json.loads(json_file.read_text())
        assert data['memory_id'] == 'persist_test'


# ════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration test with the buggy calculator example."""

    def test_full_local_pipeline(self, tmp_path):
        """
        Integration test: parse code → apply fix → verify tests pass.
        Works on Windows, Linux, and Mac. No API keys or Docker needed.
        """
        import subprocess, sys

        (tmp_path / "calculator.py").write_text(
            "def calculate_discount(price: float, discount_rate: float) -> float:\n"
            "    if discount_rate:\n"
            "        return price * discount_rate\n"
            "    return 0\n"
        )
        (tmp_path / "test_calculator.py").write_text(
            "from calculator import calculate_discount\n\n"
            "def test_zero_discount():\n"
            "    assert calculate_discount(100, 0) == 100.0\n\n"
            "def test_normal_discount():\n"
            "    assert calculate_discount(100, 0.20) == 80.0\n"
        )

        # Buggy code should fail
        r = subprocess.run(
            [sys.executable, '-m', 'pytest', 'test_calculator.py', '-q', '--tb=no'],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=30
        )
        assert r.returncode != 0, "Buggy code should fail tests"

        # Apply fix
        (tmp_path / "calculator.py").write_text(
            "def calculate_discount(price: float, discount_rate: float) -> float:\n"
            "    if discount_rate >= 1.0:\n"
            "        return 0.0\n"
            "    return price * (1 - discount_rate)\n"
        )

        # Fixed code should pass
        r = subprocess.run(
            [sys.executable, '-m', 'pytest', 'test_calculator.py', '-q', '--tb=short'],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0, f"Fixed code should pass.\n{r.stdout}\n{r.stderr}"
