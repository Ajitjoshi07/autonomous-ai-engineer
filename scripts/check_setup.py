"""
Setup & Config Checker
=======================
Run this BEFORE attempting the full pipeline.
Checks every dependency, API key, and tool — tells you exactly what's
working and what still needs to be set up.

Run: python scripts/check_setup.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / 'config' / '.env')
except ImportError:
    pass


# ── Terminal colors (works on Windows too) ───────────────────────────────────
class C:
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'

def ok(msg):   print(f"  {C.GREEN}✓{C.RESET} {msg}")
def warn(msg): print(f"  {C.YELLOW}⚠{C.RESET} {msg}")
def fail(msg): print(f"  {C.RED}✗{C.RESET} {msg}")
def info(msg): print(f"  {C.CYAN}→{C.RESET} {msg}")
def header(msg): print(f"\n{C.BOLD}{C.CYAN}{msg}{C.RESET}")
def section(msg): print(f"\n{'─'*50}\n  {C.BOLD}{msg}{C.RESET}\n{'─'*50}")


results = {'pass': 0, 'warn': 0, 'fail': 0}

def check(condition, ok_msg, fail_msg, is_warning=False):
    if condition:
        ok(ok_msg)
        results['pass'] += 1
        return True
    else:
        if is_warning:
            warn(fail_msg)
            results['warn'] += 1
        else:
            fail(fail_msg)
            results['fail'] += 1
        return False


# ════════════════════════════════════════════════════════════════════════════
section("1. Python Environment")
# ════════════════════════════════════════════════════════════════════════════

version = sys.version_info
check(
    version >= (3, 10),
    f"Python {version.major}.{version.minor}.{version.micro}",
    f"Python 3.10+ required (you have {version.major}.{version.minor})"
)

# Check if in virtual environment
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
check(
    in_venv,
    "Running inside virtual environment",
    "Not in a virtual environment — run: venv\\Scripts\\activate (Windows)",
    is_warning=True
)


# ════════════════════════════════════════════════════════════════════════════
section("2. Core Dependencies")
# ════════════════════════════════════════════════════════════════════════════

packages = [
    ('rich',               'rich',                  'Terminal UI'),
    ('langgraph',          'langgraph',              'Agent orchestration (LangGraph)'),
    ('langchain',          'langchain',              'LangChain'),
    ('openai',             'openai',                 'OpenAI SDK'),
    ('pydantic',           'pydantic',               'Data validation'),
    ('dotenv',             'python-dotenv',          'Config loading'),
    ('faiss',              'faiss-cpu',              'Vector similarity search'),
    ('sentence_transformers', 'sentence-transformers', 'Text embeddings'),
    ('networkx',           'networkx',               'Call graph'),
    ('git',                'gitpython',              'Git operations'),
    ('docker',             'docker',                 'Docker sandbox'),
    ('unidiff',            'unidiff',                'Patch parsing'),
    ('github',             'PyGithub',               'GitHub API'),
    ('pytest',             'pytest',                 'Testing framework'),
    ('ruff',               'ruff',                   'Code linter'),
]

for import_name, pip_name, description in packages:
    try:
        __import__(import_name)
        ok(f"{description} ({pip_name})")
        results['pass'] += 1
    except ImportError:
        fail(f"{description} — run: pip install {pip_name}")
        results['fail'] += 1

# Check tree-sitter separately (version-sensitive)
try:
    import tree_sitter
    import tree_sitter_python
    ok(f"Code parser (tree-sitter + tree-sitter-python)")
    results['pass'] += 1
except ImportError:
    warn(f"tree-sitter not installed — run: pip install tree-sitter tree-sitter-python")
    info("Will use Python's built-in ast module as fallback")
    results['warn'] += 1

# Check groq
try:
    import groq
    ok("Groq SDK (groq)")
    results['pass'] += 1
except ImportError:
    warn("Groq SDK not installed — run: pip install groq")
    info("Only needed if using Groq as LLM provider")
    results['warn'] += 1

# Check torch (optional, for CodeBERT)
try:
    import torch
    ok(f"PyTorch {torch.__version__} (for CodeBERT)")
    results['pass'] += 1
except ImportError:
    warn("PyTorch not installed — sentence-transformers will be used as fallback")
    info("Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    results['warn'] += 1


# ════════════════════════════════════════════════════════════════════════════
section("3. External Tools")
# ════════════════════════════════════════════════════════════════════════════

# Docker
try:
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        ok(f"Docker: {result.stdout.strip()}")
        results['pass'] += 1

        # Check Docker daemon is running
        try:
            import docker
            client = docker.from_env()
            client.ping()
            ok("Docker daemon is running")
            results['pass'] += 1
        except Exception:
            warn("Docker daemon not running — start Docker Desktop")
            info("Download: https://www.docker.com/products/docker-desktop/")
            results['warn'] += 1
    else:
        warn("Docker not found")
        results['warn'] += 1
except (FileNotFoundError, subprocess.TimeoutExpired):
    warn("Docker not installed — sandbox will use unsafe local mode")
    info("Download Docker Desktop: https://www.docker.com/products/docker-desktop/")
    results['warn'] += 1

# ruff linter
try:
    result = subprocess.run(['ruff', '--version'], capture_output=True, text=True, timeout=5)
    check(result.returncode == 0, f"ruff linter: {result.stdout.strip()}", "ruff not found — run: pip install ruff")
except FileNotFoundError:
    fail("ruff not found — run: pip install ruff")
    results['fail'] += 1


# ════════════════════════════════════════════════════════════════════════════
section("4. API Keys & Config")
# ════════════════════════════════════════════════════════════════════════════

env_file = ROOT / 'config' / '.env'
check(
    env_file.exists(),
    "config/.env file exists",
    "config/.env missing — copy from config/.env.example"
)

# LLM Provider
provider = os.getenv('LLM_PROVIDER', '')
model = os.getenv('LLM_MODEL', '')

if provider == 'groq':
    groq_key = os.getenv('GROQ_API_KEY', '')
    if groq_key and groq_key != 'gsk_your-key-here' and len(groq_key) > 10:
        ok(f"Groq API key set (provider: groq, model: {model})")
        results['pass'] += 1

        # Test the key actually works
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model=model or 'llama-3.1-8b-instant',
                messages=[{"role": "user", "content": "Say 'OK' in one word"}],
                max_tokens=5
            )
            ok(f"Groq API key WORKS ✓ (test response: {response.choices[0].message.content.strip()})")
            results['pass'] += 1
        except Exception as e:
            fail(f"Groq API key set but test failed: {e}")
            results['fail'] += 1
    else:
        fail("GROQ_API_KEY not set — get free key at console.groq.com")
        results['fail'] += 1

elif provider == 'openai':
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if openai_key and openai_key != 'sk-your-key-here' and len(openai_key) > 10:
        ok(f"OpenAI API key set (model: {model})")
        results['pass'] += 1
    else:
        fail("OPENAI_API_KEY not set")
        results['fail'] += 1

elif provider == 'ollama':
    try:
        import requests
        r = requests.get('http://localhost:11434/api/tags', timeout=3)
        ok(f"Ollama running locally (provider: ollama)")
        results['pass'] += 1
    except Exception:
        fail("Ollama not running — start with: ollama serve")
        results['fail'] += 1

else:
    warn(f"LLM_PROVIDER not set in config/.env (got: '{provider}')")
    info("Set LLM_PROVIDER=groq in config/.env")
    results['warn'] += 1

# GitHub Token
github_token = os.getenv('GITHUB_TOKEN', '')
github_username = os.getenv('GITHUB_USERNAME', '')

if github_token and github_token != 'ghp_your-token-here' and len(github_token) > 10:
    ok(f"GITHUB_TOKEN set")
    results['pass'] += 1

    # Test GitHub token
    try:
        from github import Github
        gh = Github(github_token)
        user = gh.get_user()
        ok(f"GitHub token WORKS ✓ (logged in as: {user.login})")
        results['pass'] += 1
    except Exception as e:
        fail(f"GitHub token set but invalid: {e}")
        results['fail'] += 1
else:
    warn("GITHUB_TOKEN not set — PR creation will be skipped")
    info("Get free token at: github.com/settings/tokens")
    results['warn'] += 1

if github_username and github_username != 'your-github-username':
    ok(f"GITHUB_USERNAME: {github_username}")
    results['pass'] += 1
else:
    warn("GITHUB_USERNAME not set in config/.env")
    results['warn'] += 1


# ════════════════════════════════════════════════════════════════════════════
section("5. Project Structure")
# ════════════════════════════════════════════════════════════════════════════

required_files = [
    'src/layer1_understanding/engine.py',
    'src/layer2_planning/agent.py',
    'src/layer3_sandbox/sandbox.py',
    'src/layer4_codegen/patch_engine.py',
    'src/layer5_feedback/feedback_loop.py',
    'src/layer6_critic/critic.py',
    'src/layer7_memory/memory_store.py',
    'src/orchestrator.py',
    'src/github_integration.py',
    'tests/test_all_layers.py',
    'config/.env.example',
    'requirements.txt',
]

for f in required_files:
    path = ROOT / f
    check(path.exists(), f, f"MISSING: {f}")

# Data directories
for d in ['data/memory_store', 'data/faiss_index', 'data/logs']:
    path = ROOT / d
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        ok(f"Created directory: {d}")
        results['pass'] += 1
    else:
        ok(f"Directory exists: {d}")
        results['pass'] += 1


# ════════════════════════════════════════════════════════════════════════════
section("6. Quick Layer Tests (no API keys needed)")
# ════════════════════════════════════════════════════════════════════════════

# Test Layer 1 parsing
try:
    from layer1_understanding.engine import ASTParser
    parser = ASTParser()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def hello():\n    return 42\n")
        tmp = f.name
    chunks = parser.parse_file(tmp)
    os.unlink(tmp)
    check(len(chunks) > 0, f"Layer 1: AST parser works ({len(chunks)} chunks extracted)", "Layer 1: AST parser failed")
except Exception as e:
    fail(f"Layer 1: {e}")
    results['fail'] += 1

# Test Layer 3 sandbox (local mode)
try:
    from layer3_sandbox.sandbox import DockerSandbox
    sb = DockerSandbox(timeout_seconds=5)
    sb.docker_client = None
    result_cmd = sb.run_command('echo test123')
    check('test123' in result_cmd.stdout, "Layer 3: Sandbox (local mode) works", "Layer 3: Sandbox local mode failed")
except Exception as e:
    fail(f"Layer 3: {e}")
    results['fail'] += 1

# Test Layer 7 memory
try:
    import tempfile
    from layer7_memory.memory_store import EpisodicMemoryStore
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = EpisodicMemoryStore(store_path=tmp_dir)
    ok("Layer 7: Memory store initializes correctly")
    results['pass'] += 1
except Exception as e:
    fail(f"Layer 7: {e}")
    results['fail'] += 1

# Run pytest
try:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=no'],
        capture_output=True, text=True, cwd=str(ROOT), timeout=60
    )
    passed_line = [l for l in result.stdout.split('\n') if 'passed' in l]
    if passed_line and result.returncode == 0:
        ok(f"All unit tests: {passed_line[-1].strip()}")
        results['pass'] += 1
    else:
        warn(f"Some tests failed:\n{result.stdout[-300:]}")
        results['warn'] += 1
except Exception as e:
    warn(f"Could not run pytest: {e}")
    results['warn'] += 1


# ════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*50}")
print(f"{C.BOLD}  SETUP CHECK COMPLETE{C.RESET}")
print(f"{'═'*50}")
print(f"  {C.GREEN}✓ Passed:  {results['pass']}{C.RESET}")
print(f"  {C.YELLOW}⚠ Warnings: {results['warn']}{C.RESET}")
print(f"  {C.RED}✗ Failed:  {results['fail']}{C.RESET}")
print(f"{'═'*50}")

if results['fail'] == 0 and results['warn'] == 0:
    print(f"\n  {C.GREEN}{C.BOLD}🚀 EVERYTHING READY — Run the full pipeline!{C.RESET}")
    print(f"\n  python src/orchestrator.py --local-path . --issue-text \"Your bug description\"")
elif results['fail'] == 0:
    print(f"\n  {C.YELLOW}{C.BOLD}⚠ MOSTLY READY — Fix warnings above, then run the pipeline{C.RESET}")
    print(f"\n  You can still run the demo:")
    print(f"  python scripts/demo_run.py")
else:
    print(f"\n  {C.RED}{C.BOLD}✗ NOT READY — Fix the failed items above first{C.RESET}")
    print(f"\n  Most important: set up config/.env with your API keys")
    print(f"  Free Groq key: https://console.groq.com")
    print(f"  GitHub token:  https://github.com/settings/tokens")

print()
