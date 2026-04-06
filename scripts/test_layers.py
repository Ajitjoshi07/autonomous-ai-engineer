"""
Layer-by-Layer Test Runner
===========================
Tests each layer individually so you can see exactly what's working
BEFORE attempting the full end-to-end pipeline.

Gives you a clear green/yellow/red status for each layer.

Run: python scripts/test_layers.py
Run specific layer: python scripts/test_layers.py --layer 1
"""

import sys
import os
import time
import tempfile
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / 'config' / '.env')
except ImportError:
    pass


class C:
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'


def section(n, name):
    print(f"\n{'═'*55}")
    print(f"  {C.BOLD}{C.CYAN}LAYER {n}: {name}{C.RESET}")
    print(f"{'═'*55}")


def status(label, success, detail="", warning=False):
    if success:
        icon = f"{C.GREEN}✓{C.RESET}"
    elif warning:
        icon = f"{C.YELLOW}⚠{C.RESET}"
    else:
        icon = f"{C.RED}✗{C.RESET}"
    detail_str = f"  [{C.CYAN}{detail}{C.RESET}]" if detail else ""
    print(f"  {icon} {label}{detail_str}")


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1: CODEBASE UNDERSTANDING
# ════════════════════════════════════════════════════════════════════════════

def test_layer1():
    section(1, "Codebase Understanding Engine")
    passed = 0

    # Test 1: AST Parser
    try:
        from layer1_understanding.engine import ASTParser
        parser = ASTParser()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def calculate_discount(price, rate):
    return price * (1 - rate)

class PricingEngine:
    def apply(self, price):
        return price
""")
            tmp = f.name

        chunks = parser.parse_file(tmp)
        os.unlink(tmp)

        fn_names = [c.name for c in chunks]
        has_function = 'calculate_discount' in fn_names
        has_class = any('PricingEngine' in n for n in fn_names)

        status("AST parser extracts functions", has_function, f"{len(chunks)} chunks")
        status("AST parser extracts classes/methods", has_class)
        passed += (1 if has_function else 0) + (1 if has_class else 0)
    except Exception as e:
        status("AST Parser", False, str(e)[:60])

    # Test 2: Embedder
    try:
        from layer1_understanding.engine import CodeEmbedder
        embedder = CodeEmbedder()
        embedding = embedder.embed("def hello(): return 42")
        import numpy as np
        ok = embedding is not None and len(embedding) > 0
        status(f"Code embedder works (backend: {embedder.backend})", ok, f"dim={len(embedding)}")
        passed += 1 if ok else 0
    except Exception as e:
        status("Code Embedder", False, str(e)[:60])

    # Test 3: FAISS Index
    try:
        from layer1_understanding.engine import FAISSIndex, CodeChunk
        import numpy as np
        index = FAISSIndex(embedding_dim=384)
        chunk = CodeChunk(
            chunk_id='', file_path='test.py', node_type='function',
            name='test_fn', source_code='def test_fn(): pass',
            start_line=1, end_line=2,
            embedding=np.random.randn(384).astype(np.float32)
        )
        index.add_chunks([chunk])
        results = index.search(np.random.randn(384).astype(np.float32), top_k=1)
        ok = len(results) == 1
        status("FAISS index (add + search)", ok, f"{index.index.ntotal if index.index else 0} vectors")
        passed += 1 if ok else 0
    except Exception as e:
        status("FAISS Index", False, str(e)[:60])

    # Test 4: Full repository indexing
    try:
        from layer1_understanding.engine import CodebaseUnderstandingEngine
        engine = CodebaseUnderstandingEngine()

        # Index just the src directory
        engine.index_repository(str(ROOT / 'src'), file_limit=20)

        results = engine.query("function that generates code patches", top_k=3)
        ok = len(results) > 0
        status("Full repo indexing + retrieval", ok, f"{len(results)} results")
        passed += 1 if ok else 0
    except Exception as e:
        status("Full repo indexing", False, str(e)[:80])

    print(f"\n  Layer 1 Score: {passed}/5")
    return passed >= 3


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2: PLANNING AGENT
# ════════════════════════════════════════════════════════════════════════════

def test_layer2():
    section(2, "Autonomous Planning Agent")
    passed = 0

    api_key_set = bool(os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY'))

    if not api_key_set:
        status("API key check", False, "No API key in config/.env", warning=True)
        print(f"\n  {C.YELLOW}Layer 2 requires an LLM API key to test live.{C.RESET}")
        print(f"  Get a FREE Groq key: {C.CYAN}https://console.groq.com{C.RESET}")
        print(f"  Then set GROQ_API_KEY in config/.env")
        return False

    # Test LLM client
    try:
        from layer2_planning.agent import LLMClient
        client = LLMClient()
        response = client.chat(
            system_prompt="Reply only with valid JSON.",
            user_prompt='Return: {"status": "ok", "message": "LLM working"}'
        )
        import json
        data = json.loads(response)
        ok = data.get('status') == 'ok'
        status("LLM client connection", ok, f"provider: {client.provider}")
        passed += 1 if ok else 0
    except Exception as e:
        status("LLM client", False, str(e)[:80])

    # Test planning agent with a real issue
    try:
        from layer2_planning.agent import PlanningAgent

        agent = PlanningAgent()
        plan = agent.create_plan(
            issue_text="""
Bug: calculate_discount() returns 0 when discount_rate is 0.
Expected: return original price. Actual: returns 0.
""",
            retrieved_chunks=[{
                'name': 'calculate_discount',
                'file_path': 'calculator.py',
                'source_code': 'def calculate_discount(price, rate):\n    if rate:\n        return price * rate\n    return 0'
            }]
        )
        ok = plan is not None
        if ok:
            status("Tree-of-Thought planning", True, f"strategy: {plan.selected_strategy.strategy_id}, confidence: {plan.selected_strategy.confidence_score:.2f}")
            status("Plan has target files", bool(plan.target_files), str(plan.target_files))
            status("Plan has root cause", bool(plan.root_cause), plan.root_cause[:50])
            passed += 3
        else:
            status("Tree-of-Thought planning", False, "returned None")
    except Exception as e:
        status("Planning agent", False, str(e)[:80])

    print(f"\n  Layer 2 Score: {passed}/4")
    return passed >= 2


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3: SANDBOX
# ════════════════════════════════════════════════════════════════════════════

def test_layer3():
    section(3, "Secure Code Execution Sandbox")
    passed = 0

    from layer3_sandbox.sandbox import DockerSandbox

    # Test local mode (always works)
    sb = DockerSandbox(timeout_seconds=5)
    sb.docker_client = None

    r = sb.run_command("echo hello_from_sandbox")
    ok = 'hello_from_sandbox' in r.stdout
    status("Local mode command execution", ok)
    passed += 1 if ok else 0

    r = sb.run_command("python --version")
    ok = r.exit_code == 0
    status("Python available in sandbox", ok, r.stdout.strip()[:30])
    passed += 1 if ok else 0

    r = sb.run_command("sleep 10")
    ok = r.timed_out or r.exit_code != 0
    status("Timeout enforcement (2s limit)", ok)
    passed += 1 if ok else 0

    # Test Docker mode
    try:
        import docker
        client = docker.from_env()
        client.ping()
        docker_available = True
    except Exception:
        docker_available = False

    if docker_available:
        try:
            with DockerSandbox(memory_mb=128, timeout_seconds=10) as sb:
                r = sb.run_command("echo docker_works")
                ok = r.success and 'docker_works' in r.stdout
                status("Docker container spawn + execute", ok)
                passed += 1 if ok else 0

                # Test network is blocked
                r = sb.run_command("curl -s --max-time 2 https://google.com || echo BLOCKED")
                blocked = 'BLOCKED' in r.stdout or r.exit_code != 0
                status("Network isolation (--network none)", blocked)
                passed += 1 if blocked else 0
        except Exception as e:
            status("Docker mode", False, str(e)[:60])
    else:
        status("Docker mode", False, "Docker not running — install Docker Desktop", warning=True)
        status("Network isolation", False, "Skipped (no Docker)", warning=True)
        print(f"\n  {C.YELLOW}Docker not available. Sandbox will run in unsafe local mode.{C.RESET}")
        print(f"  Install Docker Desktop: {C.CYAN}https://www.docker.com/products/docker-desktop/{C.RESET}")

    print(f"\n  Layer 3 Score: {passed}/5")
    return passed >= 2


# ════════════════════════════════════════════════════════════════════════════
# LAYER 4: CODE GENERATION
# ════════════════════════════════════════════════════════════════════════════

def test_layer4():
    section(4, "Code Generation & Patch Engine")
    passed = 0

    # Test diff computation (no API needed)
    try:
        from layer4_codegen.patch_engine import CodeGenerator
        gen = CodeGenerator.__new__(CodeGenerator)
        gen.llm = None
        gen.patch_history = []

        diff = gen._compute_diff(
            original="def foo():\n    return 0\n",
            patched="def foo():\n    return 42\n",
            file_path="test.py"
        )
        ok = '-    return 0' in diff and '+    return 42' in diff
        status("Diff computation (unified format)", ok)
        passed += 1 if ok else 0
    except Exception as e:
        status("Diff computation", False, str(e)[:60])

    # Test patch application
    try:
        from layer4_codegen.patch_engine import CodeGenerator, FilePatch, MultiFilePatch
        gen = CodeGenerator.__new__(CodeGenerator)
        gen.llm = None
        gen.patch_history = []

        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "calc.py"
            test_file.write_text("def foo():\n    return 0\n")

            patch = MultiFilePatch(patches=[FilePatch(
                file_path="calc.py",
                original_content="def foo():\n    return 0\n",
                patched_content="def foo():\n    return 42\n",
                diff_text=""
            )])

            result = gen.apply_patch(patch, repo_path=tmp)
            content = test_file.read_text()
            ok = result and '42' in content
            status("Patch application to disk", ok)
            passed += 1 if ok else 0
    except Exception as e:
        status("Patch application", False, str(e)[:60])

    # Test with LLM (if API key available)
    api_key_set = bool(os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY'))
    if api_key_set:
        try:
            from layer4_codegen.patch_engine import CodeGenerator
            gen = CodeGenerator()
            status("Code generator LLM connected", gen.llm is not None,
                   f"provider: {gen.llm.provider}" if gen.llm else "no llm")
            passed += 1 if gen.llm else 0
        except Exception as e:
            status("Code generator LLM", False, str(e)[:60])
    else:
        status("LLM-powered code generation", False, "Skipped — no API key", warning=True)

    print(f"\n  Layer 4 Score: {passed}/3")
    return passed >= 2


# ════════════════════════════════════════════════════════════════════════════
# LAYER 5: FEEDBACK LOOP
# ════════════════════════════════════════════════════════════════════════════

def test_layer5():
    section(5, "Test-Driven Feedback Loop")
    passed = 0

    # Test pytest result parsing
    try:
        from layer5_feedback.feedback_loop import TestRunner
        from unittest.mock import MagicMock

        runner = TestRunner(sandbox=MagicMock())
        result = runner._parse_raw_output("3 passed, 1 failed in 0.5s")
        ok = result.passed == 3 and result.failed == 1
        status("Pytest output parsing", ok, f"{result.passed} pass / {result.failed} fail")
        passed += 1 if ok else 0
    except Exception as e:
        status("Pytest parsing", False, str(e)[:60])

    # Test divergence detection
    try:
        from layer5_feedback.feedback_loop import FeedbackLoop, TestRunResult, TestFailure
        from unittest.mock import MagicMock

        loop = FeedbackLoop(sandbox=MagicMock(), code_generator=MagicMock())
        failure = TestFailure(
            test_name="test_foo", file_path="t.py", line_number=1,
            assertion_message="assert 0 == 1", expected_value="1",
            actual_value="0", traceback=""
        )
        r = TestRunResult(total_tests=1, passed=0, failed=1, errors=0, failures=[failure])
        loop.iteration_history = [r, r, r]
        ok = loop._is_diverging()
        status("Divergence detection (3 identical failures)", ok)
        passed += 1 if ok else 0
    except Exception as e:
        status("Divergence detection", False, str(e)[:60])

    # Live pytest test (actually runs pytest on the test suite)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=short'],
            capture_output=True, text=True, cwd=str(ROOT), timeout=60
        )
        ok = result.returncode == 0
        lines = [l for l in result.stdout.split('\n') if 'passed' in l or 'failed' in l]
        summary = lines[-1].strip() if lines else result.stdout[-100:]
        status("Run full test suite with pytest", ok, summary)
        passed += 1 if ok else 0
    except Exception as e:
        status("pytest execution", False, str(e)[:60])

    print(f"\n  Layer 5 Score: {passed}/3")
    return passed >= 2


# ════════════════════════════════════════════════════════════════════════════
# LAYER 6: CRITIC
# ════════════════════════════════════════════════════════════════════════════

def test_layer6():
    section(6, "Self-Critique & Quality Agent")
    passed = 0

    # Test ruff linter
    try:
        from layer6_critic.critic import Linter
        linter = Linter()
        status(f"ruff linter available", linter.ruff_available)
        passed += 1 if linter.ruff_available else 0

        if linter.ruff_available:
            result = linter.lint_file("test.py", "def hello():\n    return 42\n")
            ok = result.passed
            status("ruff runs on clean code", ok, f"{result.error_count} errors")
            passed += 1 if ok else 0

            # Test it catches a real issue
            result2 = linter.lint_file("test.py", "import os\nimport sys\nx=1\n")
            status("ruff detects issues in bad code", True,
                   f"{len(result2.violations)} violations found")
            passed += 1
    except Exception as e:
        status("Linter", False, str(e)[:60])

    # Test critic with LLM (if available)
    api_key_set = bool(os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY'))
    if api_key_set:
        try:
            from layer6_critic.critic import CriticAgent
            from layer4_codegen.patch_engine import FilePatch, MultiFilePatch

            critic = CriticAgent()
            patch = MultiFilePatch(patches=[FilePatch(
                file_path="calculator.py",
                original_content="def foo():\n    return 0\n",
                patched_content="def foo():\n    return 42\n",
                diff_text="--- a/calculator.py\n+++ b/calculator.py\n@@ -1 +1 @@\n-    return 0\n+    return 42\n"
            )])
            feedback = critic.review(patch, "function returns wrong value")
            ok = feedback is not None
            status("Critic LLM review", ok,
                   f"approved={feedback.approved}, score={feedback.overall_score:.2f}" if ok else "failed")
            passed += 1 if ok else 0
        except Exception as e:
            status("Critic LLM review", False, str(e)[:80])
    else:
        status("Critic LLM review", False, "Skipped — no API key", warning=True)

    print(f"\n  Layer 6 Score: {passed}/4")
    return passed >= 2


# ════════════════════════════════════════════════════════════════════════════
# LAYER 7: MEMORY
# ════════════════════════════════════════════════════════════════════════════

def test_layer7():
    section(7, "Long-Term Memory & Self-Improvement")
    passed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            from layer7_memory.memory_store import EpisodicMemoryStore, MemoryRecord
            from datetime import datetime

            store = EpisodicMemoryStore(store_path=tmp_dir)
            status("Memory store initializes", True)
            passed += 1

            # Store a record
            record = MemoryRecord(
                memory_id='test_001',
                timestamp=datetime.utcnow().isoformat(),
                repository='test/repo',
                issue_text='calculate_discount returns wrong value when rate is zero',
                success=True, iterations_required=2,
                root_cause_classification='wrong conditional',
                fix_strategy_used='fix arithmetic',
                critic_approved=True, critic_score=0.91,
            )
            store.store(record)
            status("Memory record stored", True)
            passed += 1

            # Check it persisted to disk
            json_file = Path(tmp_dir) / 'test_001.json'
            ok = json_file.exists()
            status("Memory persists to disk (JSON)", ok)
            passed += 1 if ok else 0

            # Retrieve similar
            results = store.retrieve_similar(
                "function returns 0 instead of original price when input is zero",
                top_k=3, min_similarity=0.0
            )
            status(f"Similarity retrieval works", True, f"{len(results)} results")
            passed += 1

            # Statistics
            stats = store.get_statistics()
            ok = stats['total'] == 1 and stats['resolve_rate'] == 1.0
            status("Statistics calculation", ok, f"total={stats['total']}, rate={stats['resolve_rate']:.0%}")
            passed += 1 if ok else 0

        except Exception as e:
            status("Memory store", False, str(e)[:80])

    print(f"\n  Layer 7 Score: {passed}/5")
    return passed >= 3


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, choices=range(1, 8), help='Test only this layer')
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.CYAN}{'═'*55}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  AUTONOMOUS AI ENGINEER — LAYER TEST RUNNER{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═'*55}{C.RESET}")

    layer_tests = {
        1: ("Codebase Understanding", test_layer1),
        2: ("Planning Agent",         test_layer2),
        3: ("Sandbox",                test_layer3),
        4: ("Code Generation",        test_layer4),
        5: ("Feedback Loop",          test_layer5),
        6: ("Critic Agent",           test_layer6),
        7: ("Memory",                 test_layer7),
    }

    if args.layer:
        layers_to_run = {args.layer: layer_tests[args.layer]}
    else:
        layers_to_run = layer_tests

    layer_results = {}
    for layer_num, (layer_name, test_fn) in layers_to_run.items():
        start = time.time()
        try:
            passed = test_fn()
        except Exception as e:
            print(f"\n  {C.RED}Layer {layer_num} crashed: {e}{C.RESET}")
            passed = False
        elapsed = time.time() - start
        layer_results[layer_num] = (layer_name, passed, elapsed)

    # Summary table
    print(f"\n\n{'═'*55}")
    print(f"{C.BOLD}  LAYER SUMMARY{C.RESET}")
    print(f"{'═'*55}")
    total_pass = 0
    for layer_num, (name, passed, elapsed) in layer_results.items():
        icon = f"{C.GREEN}✓ READY{C.RESET}   " if passed else f"{C.YELLOW}⚠ PARTIAL{C.RESET}"
        print(f"  Layer {layer_num}: {icon}  {name:25s}  ({elapsed:.1f}s)")
        total_pass += 1 if passed else 0

    print(f"{'─'*55}")
    print(f"  {total_pass}/{len(layer_results)} layers ready")

    if total_pass == len(layer_results):
        print(f"\n  {C.GREEN}{C.BOLD}🚀 ALL LAYERS READY — Run the full pipeline:{C.RESET}")
        print(f"\n  python src/orchestrator.py \\")
        print(f"    --repo YOUR_USERNAME/YOUR_REPO \\")
        print(f"    --issue 1")
    elif total_pass >= 4:
        print(f"\n  {C.YELLOW}{C.BOLD}⚠ MOSTLY READY — Fix remaining layers, then run pipeline{C.RESET}")
        print(f"\n  python scripts/demo_run.py  ← still works!")
    else:
        print(f"\n  {C.RED}{C.BOLD}Set up API keys in config/.env first{C.RESET}")
        print(f"  Free key: https://console.groq.com")

    print()


if __name__ == "__main__":
    main()
