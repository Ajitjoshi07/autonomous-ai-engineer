# 🔑 Setup Guide — Getting Your API Keys (Free Options First!)

You have a GitHub account. Here's everything else you need, starting with FREE options.

---

## Step 1: GitHub Personal Access Token (FREE — 5 minutes)

This lets the agent read your repos and create PRs.

1. Go to: **https://github.com/settings/tokens**
2. Click **"Generate new token (classic)"**
3. Name it: `autonomous-ai-engineer`
4. Set expiration: 90 days
5. Check these scopes:
   - ✅ `repo` (Full control of private repos)
   - ✅ `read:org` (Read org data)
6. Click **"Generate token"**
7. **COPY IT NOW** — you won't see it again!
8. Paste it into `config/.env` as `GITHUB_TOKEN=ghp_...`

---

## Step 2: Get a FREE LLM API Key

You have 3 free options — start with **Groq** (fastest and free):

### Option A: Groq (RECOMMENDED — Free tier, very fast)
1. Go to: **https://console.groq.com**
2. Sign up with GitHub account (easy!)
3. Click "API Keys" → "Create API Key"
4. Copy key → paste as `GROQ_API_KEY=gsk_...` in `.env`
5. Also set: `LLM_PROVIDER=groq`

**Free tier:** ~14,400 requests/day with Llama 3.1 70B model  
**Speed:** 10x faster than OpenAI  
**Cost:** FREE

### Option B: Google Gemini (Free tier)
1. Go to: **https://aistudio.google.com**
2. Sign in with Google
3. Click "Get API Key"
4. Set `GEMINI_API_KEY=...` and `LLM_PROVIDER=gemini`

### Option C: Ollama (Completely FREE, runs locally)
1. Download from: **https://ollama.ai**
2. Run: `ollama pull llama3.1` (downloads ~4GB)
3. Set `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.1`
4. Start Ollama before running the agent

### Option D: OpenAI (Paid, best quality)
- Go to: **https://platform.openai.com/api-keys**
- Add credits (~$5 gets you 1000+ runs)
- Set `OPENAI_API_KEY=sk-...` and `LLM_PROVIDER=openai`

---

## Step 3: Install Docker (FREE — for the sandbox)

The sandbox needs Docker to safely run generated code.

1. Download **Docker Desktop** from: **https://docker.com/products/docker-desktop**
2. Install and start it
3. Test: run `docker run hello-world` in terminal

---

## Step 4: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# Install everything
pip install -r requirements.txt
```

---

## Step 5: Fill in Your Config

```bash
# Copy the example config
cp config/.env.example config/.env

# Open config/.env and fill in:
# GROQ_API_KEY=gsk_your-key-here
# GITHUB_TOKEN=ghp_your-token-here
# GITHUB_USERNAME=your-username
```

---

## Step 6: Run the Demo!

```bash
python scripts/demo_run.py
```

This runs the full 7-layer pipeline on a synthetic bug — no API key needed for the demo!

---

## Step 7: Push to GitHub

```bash
# Create a new repo on github.com first, then:
git init
git add .
git commit -m "Initial: Autonomous AI Engineer — Phase 1 scaffolding"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/autonomous-ai-engineer.git
git push -u origin main
```

---

## Cost Estimate (if using paid APIs)

| Scenario | Cost |
|----------|------|
| Development & testing (Groq) | FREE |
| 100 SWE-bench issues (GPT-4o) | ~$15 |
| Full SWE-bench eval (300 issues) | ~$45 |
| Daily development | ~$1-2 |

**Recommendation:** Use Groq free tier for all development.  
Switch to GPT-4o only for final benchmarking.

---

## Troubleshooting

**"Docker not found"** → Install Docker Desktop, make sure it's running

**"Module not found"** → Activate venv: `source venv/bin/activate`

**"API key invalid"** → Check config/.env — no quotes around keys!

**"Groq rate limit"** → Free tier has limits; add 1-second sleep between requests
