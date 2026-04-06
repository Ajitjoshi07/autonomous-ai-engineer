# Autonomous AI Software Engineer
**by Ajit Mukund Joshi** | B.Tech AI & Data Science | 12-Month Capstone

> An AI agent that reads GitHub issues, understands codebases, writes fixes, verifies them, and opens Pull Requests — with zero human intervention.

---

## Quick Start

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/autonomous-ai-engineer
cd autonomous-ai-engineer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in your config
cp config/.env.example config/.env

# 5. Run the demo
python scripts/demo_run.py
```

---

## Project Structure

```
autonomous-ai-engineer/
├── src/
│   ├── layer1_understanding/   # AST parsing, CodeBERT embeddings, FAISS index
│   ├── layer2_planning/        # Tree-of-Thought planning agent (LangGraph)
│   ├── layer3_sandbox/         # Docker execution sandbox
│   ├── layer4_codegen/         # Code generation & patch engine
│   ├── layer5_feedback/        # Test execution & retry loop
│   ├── layer6_critic/          # Self-critique & quality review
│   └── layer7_memory/          # Long-term episodic memory
├── tests/                      # Unit tests per layer
├── scripts/                    # Utility scripts (demo, eval, setup)
├── config/                     # .env, settings
└── docs/                       # Architecture diagrams, notes
```

---

## Architecture (7 Layers)

```
GitHub Issue
     │
     ▼
[L1] Codebase Understanding  →  tree-sitter + CodeBERT + FAISS
     │
     ▼
[L2] Planning Agent          →  LangGraph + Tree-of-Thought (GPT-4o)
     │
     ▼
[L3] Sandbox                 →  Docker (--network none, memory limits)
     │
     ▼
[L4] Code Generation         →  Unified diff patches (GPT-4o)
     │
     ▼
[L5] Test Feedback Loop      →  pytest + retry (up to 8x)
     │
     ▼
[L6] Self-Critique           →  Adversarial reviewer LLM
     │
     ▼
[L7] Memory                  →  FAISS episodic store → self-improvement
     │
     ▼
GitHub Pull Request ✅
```

---

## Benchmark Target
- **SWE-bench Lite**: 20%+ resolve rate (300 real GitHub issues)
- Comparable to: Devin (Cognition AI, $175M funded), SWE-agent (Princeton)
