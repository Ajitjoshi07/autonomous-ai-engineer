"""
GitHub App Webhook Server
==========================
A FastAPI server that listens for GitHub webhook events and automatically
triggers the Autonomous AI Engineer pipeline.

HOW IT WORKS:
  1. Someone labels a GitHub issue with "agent-fix"
  2. GitHub sends a POST request to this server
  3. Server validates the webhook signature
  4. Server queues the task and runs the full 7-layer pipeline
  5. Agent fixes the bug and opens a Pull Request — automatically

SETUP:
  1. Run this server:          python src/webhook_server.py
  2. Expose via ngrok:         ngrok http 8000
  3. Set webhook URL on GitHub: https://YOUR_NGROK_URL/webhook
  4. Label any issue "agent-fix" and watch the agent work

ENDPOINTS:
  POST /webhook     GitHub webhook receiver
  GET  /health      Health check
  GET  /status      Queue and task status
  GET  /tasks       All completed/running tasks
"""

import os
import sys
import hmac
import hashlib
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
load_dotenv(ROOT / 'config' / '.env')

# Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("webhook")
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
WEBHOOK_SECRET   = os.getenv('GITHUB_WEBHOOK_SECRET', '')
TRIGGER_LABEL    = os.getenv('AGENT_TRIGGER_LABEL', 'agent-fix')
MAX_QUEUE_SIZE   = int(os.getenv('MAX_QUEUE_SIZE', '10'))
GITHUB_USERNAME  = os.getenv('GITHUB_USERNAME', '')

# ── Task State ────────────────────────────────────────────────────────────────

@dataclass
class AgentTask:
    """Represents one agent task triggered by a GitHub issue."""
    task_id: str
    repo_name: str
    issue_number: int
    issue_title: str
    issue_body: str
    triggered_at: str
    status: str = "queued"          # queued | running | success | failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    iterations_used: int = 0
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'repo': self.repo_name,
            'issue': f"#{self.issue_number}: {self.issue_title}",
            'status': self.status,
            'triggered_at': self.triggered_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'pr_url': self.pr_url,
            'pr_number': self.pr_number,
            'iterations_used': self.iterations_used,
            'error': self.error,
        }


# ── App State ─────────────────────────────────────────────────────────────────
task_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
all_tasks: dict[str, AgentTask] = {}
worker_running: bool = False

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous AI Engineer — Webhook Server",
    description="Listens for GitHub issue events and triggers the AI agent pipeline",
    version="1.0.0"
)


# ── Webhook Signature Verification ────────────────────────────────────────────

def verify_github_signature(payload: bytes, signature: str) -> bool:
    """
    Verify that the webhook came from GitHub using HMAC-SHA256.
    This prevents anyone else from triggering your agent.
    """
    if not WEBHOOK_SECRET:
        log.warning("GITHUB_WEBHOOK_SECRET not set — skipping signature verification")
        return True

    if not signature or not signature.startswith('sha256='):
        return False

    expected = 'sha256=' + hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ── Webhook Endpoint ──────────────────────────────────────────────────────────

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Main webhook endpoint. GitHub sends all events here.
    
    Triggers the agent when:
    - An issue is labeled with 'agent-fix'
    - A comment says '/agent-fix' (coming in Phase 3)
    """
    # Read raw body for signature verification
    body = await request.body()

    # Verify signature
    signature = request.headers.get('X-Hub-Signature-256', '')
    if not verify_github_signature(body, signature):
        log.warning("Invalid webhook signature — rejected")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse event type
    event_type = request.headers.get('X-GitHub-Event', '')
    delivery_id = request.headers.get('X-GitHub-Delivery', 'unknown')

    import json
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    log.info(f"Received event: {event_type} (delivery: {delivery_id[:8]})")

    # ── Handle issue labeled event ─────────────────────────────────────────
    if event_type == 'issues':
        action = payload.get('action', '')

        if action == 'labeled':
            label_name = payload.get('label', {}).get('name', '')

            if label_name == TRIGGER_LABEL:
                issue = payload['issue']
                repo = payload['repository']

                log.info(f"Trigger label '{TRIGGER_LABEL}' added to issue #{issue['number']}: {issue['title']}")

                # Check queue capacity
                if task_queue.full():
                    log.warning("Task queue is full — rejecting new task")
                    return JSONResponse(
                        status_code=429,
                        content={"message": "Queue full, try again later"}
                    )

                # Create task
                task_id = f"{repo['full_name'].replace('/', '-')}-{issue['number']}-{int(time.time())}"
                task = AgentTask(
                    task_id=task_id,
                    repo_name=repo['full_name'],
                    issue_number=issue['number'],
                    issue_title=issue['title'],
                    issue_body=issue.get('body', ''),
                    triggered_at=datetime.utcnow().isoformat(),
                )
                all_tasks[task_id] = task

                # Queue the task for background processing
                await task_queue.put(task)

                log.info(f"Task queued: {task_id}")

                # Post a comment on the issue to acknowledge
                background_tasks.add_task(
                    post_acknowledgement_comment,
                    repo_name=repo['full_name'],
                    issue_number=issue['number'],
                    task_id=task_id
                )

                return JSONResponse(content={
                    "message": "Task queued",
                    "task_id": task_id,
                    "issue": f"#{issue['number']}: {issue['title']}"
                })

    # Acknowledge all other events
    return JSONResponse(content={"message": f"Event '{event_type}' received but not processed"})


# ── Health & Status Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by monitoring tools."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "queue_size": task_queue.qsize(),
        "total_tasks": len(all_tasks),
        "worker_running": worker_running,
        "trigger_label": TRIGGER_LABEL,
    }


@app.get("/status")
async def status():
    """Current queue and worker status."""
    running = [t for t in all_tasks.values() if t.status == 'running']
    queued = [t for t in all_tasks.values() if t.status == 'queued']
    completed = [t for t in all_tasks.values() if t.status == 'success']
    failed = [t for t in all_tasks.values() if t.status == 'failed']

    return {
        "worker_running": worker_running,
        "queue_depth": task_queue.qsize(),
        "running": len(running),
        "queued": len(queued),
        "completed": len(completed),
        "failed": len(failed),
        "current_task": running[0].to_dict() if running else None,
    }


@app.get("/tasks")
async def list_tasks():
    """List all tasks with their status."""
    return {
        "tasks": [t.to_dict() for t in sorted(
            all_tasks.values(),
            key=lambda t: t.triggered_at,
            reverse=True
        )],
        "total": len(all_tasks)
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get details of a specific task."""
    task = all_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    result = task.to_dict()
    result['logs'] = task.logs[-50:]  # Last 50 log lines
    return result


# ── Background Worker ─────────────────────────────────────────────────────────

async def run_agent_worker():
    """
    Background worker that processes tasks from the queue.
    Runs the full 7-layer pipeline for each task.
    Processes tasks one at a time to avoid resource conflicts.
    """
    global worker_running
    log.info("Agent worker started — waiting for tasks...")

    while True:
        try:
            # Wait for next task (blocks until one arrives)
            task: AgentTask = await task_queue.get()
            worker_running = True

            log.info(f"Starting task: {task.task_id}")
            log.info(f"Repo: {task.repo_name} | Issue: #{task.issue_number}")

            task.status = 'running'
            task.started_at = datetime.utcnow().isoformat()
            task.logs.append(f"[{task.started_at}] Task started")

            try:
                # Run the pipeline in a thread pool to avoid blocking the event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_pipeline_sync,
                    task
                )

                if result.get('success'):
                    task.status = 'success'
                    task.pr_url = result.get('pr_url')
                    task.pr_number = result.get('pr_number')
                    task.iterations_used = result.get('iterations', 0)
                    log.info(f"Task SUCCESS: PR opened at {task.pr_url}")
                    task.logs.append(f"SUCCESS: PR opened at {task.pr_url}")
                else:
                    task.status = 'failed'
                    task.error = result.get('error', 'Unknown error')
                    log.error(f"Task FAILED: {task.error}")
                    task.logs.append(f"FAILED: {task.error}")

            except Exception as e:
                task.status = 'failed'
                task.error = str(e)
                log.error(f"Task crashed: {e}")
                task.logs.append(f"CRASHED: {e}")

            finally:
                task.completed_at = datetime.utcnow().isoformat()
                task_queue.task_done()
                worker_running = False

                elapsed = ""
                if task.started_at:
                    start = datetime.fromisoformat(task.started_at)
                    end = datetime.fromisoformat(task.completed_at)
                    elapsed = f" ({(end-start).seconds}s)"

                log.info(f"Task {task.status}{elapsed}: {task.task_id}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Worker error: {e}")
            worker_running = False
            await asyncio.sleep(5)


def run_pipeline_sync(task: AgentTask) -> dict:
    """
    Runs the orchestrator pipeline synchronously.
    Called in a thread pool from the async worker.
    """
    try:
        from orchestrator import AutonomousEngineer

        def log_capture(msg):
            task.logs.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

        engineer = AutonomousEngineer()
        result = engineer.run(
            issue_text=f"Title: {task.issue_title}\n\n{task.issue_body}",
            repo_name=task.repo_name,
            issue_number=task.issue_number
        )
        return result

    except Exception as e:
        return {'success': False, 'error': str(e)}


async def post_acknowledgement_comment(repo_name: str, issue_number: int, task_id: str):
    """Post a comment on the issue to let the user know the agent is working."""
    try:
        from github import Github, Auth
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            return

        gh = Github(auth=Auth.Token(token))
        repo = gh.get_repo(repo_name)
        issue = repo.get_issue(issue_number)

        issue.create_comment(
            f"🤖 **Autonomous AI Engineer** has picked up this issue.\n\n"
            f"**Task ID:** `{task_id}`\n\n"
            f"I am now:\n"
            f"1. 🔍 Indexing the codebase\n"
            f"2. 🧠 Generating a repair plan\n"
            f"3. ⚙️ Writing and testing a fix\n"
            f"4. 🔎 Running code review\n"
            f"5. 🚀 Opening a Pull Request\n\n"
            f"I'll update this issue when the PR is ready. "
            f"Check status at `/status` on the webhook server.\n\n"
            f"*Developed by Ajit Mukund Joshi — Autonomous AI Software Engineer*"
        )
        log.info(f"Posted acknowledgement comment on #{issue_number}")
    except Exception as e:
        log.warning(f"Could not post comment: {e}")


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Start the background worker when the server starts."""
    asyncio.create_task(run_agent_worker())
    log.info("="*55)
    log.info("  Autonomous AI Engineer — Webhook Server")
    log.info("="*55)
    log.info(f"  Trigger label : '{TRIGGER_LABEL}'")
    log.info(f"  GitHub user   : {GITHUB_USERNAME}")
    log.info(f"  Webhook secret: {'set' if WEBHOOK_SECRET else 'NOT SET (insecure)'}")
    log.info(f"  Queue capacity: {MAX_QUEUE_SIZE}")
    log.info("")
    log.info("  Endpoints:")
    log.info("    POST /webhook  — GitHub sends events here")
    log.info("    GET  /health   — Health check")
    log.info("    GET  /status   — Current queue status")
    log.info("    GET  /tasks    — All tasks")
    log.info("")
    log.info("  Next step: expose with ngrok")
    log.info("    ngrok http 8000")
    log.info("="*55)


@app.on_event("shutdown")
async def shutdown():
    log.info("Webhook server shutting down...")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous AI Engineer Webhook Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--reload', action='store_true', help='Auto-reload on code changes')
    args = parser.parse_args()

    uvicorn.run(
        "webhook_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
