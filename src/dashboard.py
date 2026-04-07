"""
Observability Dashboard Server
================================
Serves a real-time browser dashboard showing:
  - Agent task queue and status
  - Live logs per task
  - Layer-by-layer progress visualization
  - Memory store statistics
  - System health (API keys, Docker, etc.)

Run: python src/dashboard.py
Open: http://localhost:8080
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
load_dotenv(ROOT / 'config' / '.env')

logging.basicConfig(level=logging.WARNING)

app = FastAPI(title="Autonomous AI Engineer — Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous AI Engineer — Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

  :root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --border:   #1e2d45;
    --accent:   #3b82f6;
    --green:    #10b981;
    --yellow:   #f59e0b;
    --red:      #ef4444;
    --purple:   #8b5cf6;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Inter', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Header ── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
  }

  .logo-text h1 {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }

  .logo-text p {
    font-size: 11px;
    color: var(--muted);
    font-weight: 300;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--green);
    font-weight: 600;
  }

  .live-dot {
    width: 8px;
    height: 8px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  /* ── Layout ── */
  .main {
    padding: 24px 32px;
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }

  .card-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Stat Cards ── */
  .stats-row {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
  }

  .stat-card:hover { border-color: var(--accent); }

  .stat-value {
    font-size: 36px;
    font-weight: 700;
    font-family: var(--mono);
    line-height: 1;
    margin-bottom: 6px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--muted);
    font-weight: 400;
  }

  .stat-green  .stat-value { color: var(--green); }
  .stat-blue   .stat-value { color: var(--accent); }
  .stat-yellow .stat-value { color: var(--yellow); }
  .stat-red    .stat-value { color: var(--red); }
  .stat-purple .stat-value { color: var(--purple); }

  /* ── Layer Pipeline ── */
  .pipeline {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 0;
  }

  .layer-node {
    flex: 1;
    text-align: center;
    position: relative;
  }

  .layer-node:not(:last-child)::after {
    content: '';
    position: absolute;
    right: -1px;
    top: 50%;
    transform: translateY(-50%);
    width: 2px;
    height: 32px;
    background: var(--border);
    z-index: 1;
  }

  .layer-node.active::after { background: var(--accent); }
  .layer-node.done::after   { background: var(--green); }

  .layer-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 8px;
    font-size: 18px;
    transition: all 0.3s;
  }

  .layer-node.active .layer-circle {
    border-color: var(--accent);
    background: rgba(59,130,246,0.15);
    animation: glow 1.5s infinite;
  }

  .layer-node.done .layer-circle {
    border-color: var(--green);
    background: rgba(16,185,129,0.15);
  }

  .layer-node.error .layer-circle {
    border-color: var(--red);
    background: rgba(239,68,68,0.15);
  }

  @keyframes glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(59,130,246,0); }
  }

  .layer-name {
    font-size: 10px;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .layer-node.active .layer-name { color: var(--accent); }
  .layer-node.done .layer-name   { color: var(--green); }

  /* ── Task List ── */
  .task-list { display: flex; flex-direction: column; gap: 10px; }

  .task-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s;
  }

  .task-item:hover { border-color: var(--accent); }

  .task-status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
  }

  .dot-queued  { background: var(--muted); }
  .dot-running { background: var(--accent); animation: pulse 1.5s infinite; }
  .dot-success { background: var(--green); }
  .dot-failed  { background: var(--red); }

  .task-info { flex: 1; min-width: 0; }
  .task-title { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .task-meta  { font-size: 11px; color: var(--muted); margin-top: 3px; }

  .task-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    flex-shrink: 0;
  }

  .badge-queued  { background: rgba(100,116,139,0.2); color: var(--muted); }
  .badge-running { background: rgba(59,130,246,0.2);  color: var(--accent); }
  .badge-success { background: rgba(16,185,129,0.2);  color: var(--green); }
  .badge-failed  { background: rgba(239,68,68,0.2);   color: var(--red); }

  /* ── Log Terminal ── */
  .log-terminal {
    background: #050810;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    height: 280px;
    overflow-y: auto;
    color: #94a3b8;
  }

  .log-terminal::-webkit-scrollbar { width: 4px; }
  .log-terminal::-webkit-scrollbar-track { background: transparent; }
  .log-terminal::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .log-line { display: block; }
  .log-time  { color: var(--muted); }
  .log-info  { color: #93c5fd; }
  .log-ok    { color: var(--green); }
  .log-warn  { color: var(--yellow); }
  .log-error { color: var(--red); }

  /* ── Memory Stats ── */
  .memory-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .memory-stat {
    background: var(--bg);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }

  .memory-stat-value {
    font-size: 24px;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--purple);
  }

  .memory-stat-label {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }

  /* ── Health Indicators ── */
  .health-list { display: flex; flex-direction: column; gap: 8px; }

  .health-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--bg);
    border-radius: 8px;
    font-size: 13px;
  }

  .health-ok     { color: var(--green); font-weight: 600; font-size: 12px; }
  .health-warn   { color: var(--yellow); font-weight: 600; font-size: 12px; }
  .health-error  { color: var(--red); font-weight: 600; font-size: 12px; }

  /* ── PR Link ── */
  .pr-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--accent);
    text-decoration: none;
    font-family: var(--mono);
  }

  .pr-link:hover { text-decoration: underline; }

  /* ── Empty State ── */
  .empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--muted);
    font-size: 13px;
  }

  .empty-icon { font-size: 36px; margin-bottom: 12px; }

  /* ── Resolve rate bar ── */
  .progress-bar {
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 8px;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--green));
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .wide { grid-column: span 2; }
  .full { grid-column: 1 / -1; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🤖</div>
    <div class="logo-text">
      <h1>Autonomous AI Engineer</h1>
      <p>by Ajit Mukund Joshi — B.Tech AI & Data Science</p>
    </div>
  </div>
  <div class="header-right">
    <div class="live-badge">
      <div class="live-dot"></div>
      LIVE
    </div>
    <span style="font-size:12px;color:var(--muted)" id="last-update">Connecting...</span>
  </div>
</header>

<div class="main">

  <!-- Stats Row -->
  <div class="stats-row">
    <div class="stat-card stat-blue">
      <div class="stat-value" id="stat-total">0</div>
      <div class="stat-label">Total Tasks</div>
    </div>
    <div class="stat-card stat-green">
      <div class="stat-value" id="stat-success">0</div>
      <div class="stat-label">Resolved</div>
    </div>
    <div class="stat-card stat-yellow">
      <div class="stat-value" id="stat-running">0</div>
      <div class="stat-label">Running</div>
    </div>
    <div class="stat-card stat-red">
      <div class="stat-value" id="stat-failed">0</div>
      <div class="stat-label">Failed</div>
    </div>
    <div class="stat-card stat-purple">
      <div class="stat-value" id="stat-rate">0%</div>
      <div class="stat-label">Resolve Rate</div>
    </div>
  </div>

  <!-- Layer Pipeline -->
  <div class="card full">
    <div class="card-title">🔄 Agent Pipeline</div>
    <div class="pipeline" id="pipeline">
      <div class="layer-node idle" id="layer-1">
        <div class="layer-circle">🔍</div>
        <div class="layer-name">L1 Understand</div>
      </div>
      <div class="layer-node idle" id="layer-2">
        <div class="layer-circle">🧠</div>
        <div class="layer-name">L2 Plan</div>
      </div>
      <div class="layer-node idle" id="layer-3">
        <div class="layer-circle">🐳</div>
        <div class="layer-name">L3 Sandbox</div>
      </div>
      <div class="layer-node idle" id="layer-4">
        <div class="layer-circle">⚙️</div>
        <div class="layer-name">L4 Generate</div>
      </div>
      <div class="layer-node idle" id="layer-5">
        <div class="layer-circle">🧪</div>
        <div class="layer-name">L5 Test</div>
      </div>
      <div class="layer-node idle" id="layer-6">
        <div class="layer-circle">🔎</div>
        <div class="layer-name">L6 Critic</div>
      </div>
      <div class="layer-node idle" id="layer-7">
        <div class="layer-circle">💾</div>
        <div class="layer-name">L7 Memory</div>
      </div>
      <div class="layer-node idle" id="layer-pr">
        <div class="layer-circle">🚀</div>
        <div class="layer-name">PR Open</div>
      </div>
    </div>
  </div>

  <!-- Task List -->
  <div class="card wide">
    <div class="card-title">📋 Task Queue</div>
    <div class="task-list" id="task-list">
      <div class="empty-state">
        <div class="empty-icon">⏳</div>
        No tasks yet. Label a GitHub issue with <strong>agent-fix</strong> to trigger the agent.
      </div>
    </div>
  </div>

  <!-- Live Logs -->
  <div class="card">
    <div class="card-title">📟 Live Logs</div>
    <div class="log-terminal" id="log-terminal">
      <span class="log-line"><span class="log-time">[--:--:--]</span> <span class="log-info">Waiting for tasks...</span></span>
    </div>
  </div>

  <!-- Memory Stats -->
  <div class="card">
    <div class="card-title">🧠 Memory Store (Layer 7)</div>
    <div class="memory-grid">
      <div class="memory-stat">
        <div class="memory-stat-value" id="mem-total">0</div>
        <div class="memory-stat-label">Memories Stored</div>
      </div>
      <div class="memory-stat">
        <div class="memory-stat-value" id="mem-rate">0%</div>
        <div class="memory-stat-label">Resolve Rate</div>
      </div>
      <div class="memory-stat">
        <div class="memory-stat-value" id="mem-iter">0</div>
        <div class="memory-stat-label">Avg Iterations</div>
      </div>
      <div class="memory-stat">
        <div class="memory-stat-value" id="mem-cost">$0</div>
        <div class="memory-stat-label">Avg Cost/Task</div>
      </div>
    </div>
    <div class="progress-bar" style="margin-top:16px">
      <div class="progress-fill" id="mem-progress" style="width:0%"></div>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-top:6px;text-align:center" id="mem-label">
      0 / 0 issues resolved
    </div>
  </div>

  <!-- System Health -->
  <div class="card">
    <div class="card-title">💊 System Health</div>
    <div class="health-list" id="health-list">
      <div class="health-item">
        <span>Loading...</span>
      </div>
    </div>
  </div>

  <!-- Current PR -->
  <div class="card wide" id="current-pr-card" style="display:none">
    <div class="card-title">🚀 Latest Pull Request</div>
    <div id="current-pr-content"></div>
  </div>

</div>

<script>
// ── Data fetching ─────────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    return await r.json();
  } catch(e) { return null; }
}

async function fetchTasks() {
  try {
    const r = await fetch('/api/tasks');
    return await r.json();
  } catch(e) { return { tasks: [] }; }
}

async function fetchMemory() {
  try {
    const r = await fetch('/api/memory');
    return await r.json();
  } catch(e) { return {}; }
}

async function fetchHealth() {
  try {
    const r = await fetch('/api/health');
    return await r.json();
  } catch(e) { return {}; }
}

// ── Render functions ─────────────────────────────────────────────────────────

function renderStats(tasks) {
  const total   = tasks.length;
  const success = tasks.filter(t => t.status === 'success').length;
  const running = tasks.filter(t => t.status === 'running').length;
  const failed  = tasks.filter(t => t.status === 'failed').length;
  const rate    = total > 0 ? Math.round(success / total * 100) : 0;

  document.getElementById('stat-total').textContent   = total;
  document.getElementById('stat-success').textContent = success;
  document.getElementById('stat-running').textContent = running;
  document.getElementById('stat-failed').textContent  = failed;
  document.getElementById('stat-rate').textContent    = rate + '%';
}

function renderPipeline(status) {
  const layers = ['layer-1','layer-2','layer-3','layer-4','layer-5','layer-6','layer-7','layer-pr'];
  const currentLayer = status?.current_layer || 0;

  layers.forEach((id, i) => {
    const el = document.getElementById(id);
    el.className = 'layer-node';
    if (i + 1 < currentLayer) el.classList.add('done');
    else if (i + 1 === currentLayer) el.classList.add('active');
    else el.classList.add('idle');
  });
}

function renderTasks(tasks) {
  const container = document.getElementById('task-list');
  if (tasks.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⏳</div>
        No tasks yet. Label a GitHub issue with <strong>agent-fix</strong> to trigger the agent.
      </div>`;
    return;
  }

  container.innerHTML = tasks.slice(0, 10).map(task => {
    const badgeClass = `badge-${task.status}`;
    const dotClass   = `dot-${task.status}`;
    const meta = task.started_at
      ? `${task.repo} · Started ${new Date(task.started_at).toLocaleTimeString()}`
      : `${task.repo} · Queued ${new Date(task.triggered_at).toLocaleTimeString()}`;
    const prLink = task.pr_url
      ? `<a class="pr-link" href="${task.pr_url}" target="_blank">🔗 View PR #${task.pr_number}</a>`
      : '';

    return `
      <div class="task-item">
        <div class="task-status-dot ${dotClass}"></div>
        <div class="task-info">
          <div class="task-title">${task.issue}</div>
          <div class="task-meta">${meta} ${task.iterations_used ? `· ${task.iterations_used} iterations` : ''}</div>
          ${prLink}
        </div>
        <div class="task-badge ${badgeClass}">${task.status}</div>
      </div>`;
  }).join('');
}

function renderLogs(tasks) {
  const terminal = document.getElementById('log-terminal');
  const running  = tasks.find(t => t.status === 'running');
  const recent   = tasks.find(t => t.status === 'success' || t.status === 'failed');
  const task     = running || recent;

  if (!task || !task.logs || task.logs.length === 0) return;

  const lines = task.logs.slice(-30).map(line => {
    let cls = 'log-info';
    if (line.includes('SUCCESS') || line.includes('✅') || line.includes('[OK]')) cls = 'log-ok';
    else if (line.includes('FAILED') || line.includes('✗') || line.includes('ERROR')) cls = 'log-error';
    else if (line.includes('⚠') || line.includes('WARN')) cls = 'log-warn';

    const timeMatch = line.match(/[[](\d\d:\d\d:\d\d)[]]/);
    const time = timeMatch ? `<span class="log-time">[${timeMatch[1]}]</span> ` : '';
    const msg  = line.replace(/[[]\d{2}:\d{2}:\d{2}[]]\s*/, '');
    return `<span class="log-line">${time}<span class="${cls}">${msg}</span></span>`;
  }).join('\n');

  terminal.innerHTML = lines;
  terminal.scrollTop = terminal.scrollHeight;
}

function renderMemory(mem) {
  if (!mem || !mem.total) return;
  document.getElementById('mem-total').textContent = mem.total || 0;
  document.getElementById('mem-rate').textContent  = mem.resolve_rate ? Math.round(mem.resolve_rate * 100) + '%' : '0%';
  document.getElementById('mem-iter').textContent  = mem.avg_iterations ? mem.avg_iterations.toFixed(1) : '0';
  document.getElementById('mem-cost').textContent  = mem.avg_cost_usd ? '$' + mem.avg_cost_usd.toFixed(2) : '$0';

  const rate = mem.resolve_rate || 0;
  document.getElementById('mem-progress').style.width = (rate * 100) + '%';
  document.getElementById('mem-label').textContent =
    `${mem.successful || 0} / ${mem.total || 0} issues resolved`;
}

function renderHealth(health) {
  const container = document.getElementById('health-list');
  if (!health || !health.checks) return;

  container.innerHTML = health.checks.map(c => {
    const cls = c.ok ? 'health-ok' : (c.warning ? 'health-warn' : 'health-error');
    const icon = c.ok ? '✓' : (c.warning ? '⚠' : '✗');
    return `
      <div class="health-item">
        <span>${c.name}</span>
        <span class="${cls}">${icon} ${c.detail || ''}</span>
      </div>`;
  }).join('');
}

function renderLatestPR(tasks) {
  const withPR = tasks.filter(t => t.pr_url);
  if (withPR.length === 0) {
    document.getElementById('current-pr-card').style.display = 'none';
    return;
  }
  const latest = withPR[0];
  document.getElementById('current-pr-card').style.display = 'block';
  document.getElementById('current-pr-content').innerHTML = `
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
      <div>
        <div style="font-size:14px;font-weight:600">${latest.issue}</div>
        <div style="font-size:12px;color:var(--muted);margin-top:4px">${latest.repo}</div>
      </div>
      <a class="pr-link" href="${latest.pr_url}" target="_blank" style="font-size:14px">
        🔗 Pull Request #${latest.pr_number}
      </a>
      <div style="font-size:12px;color:var(--muted)">
        ${latest.iterations_used} iteration(s) · Critic approved ✅
      </div>
    </div>`;
}

// ── Main update loop ─────────────────────────────────────────────────────────

async function update() {
  const [statusData, tasksData, memData, healthData] = await Promise.all([
    fetchStatus(), fetchTasks(), fetchMemory(), fetchHealth()
  ]);

  const tasks = tasksData?.tasks || [];

  renderStats(tasks);
  renderPipeline(statusData);
  renderTasks(tasks);
  renderLogs(tasks);
  renderMemory(memData);
  renderHealth(healthData);
  renderLatestPR(tasks);

  document.getElementById('last-update').textContent =
    'Updated ' + new Date().toLocaleTimeString();
}

// Poll every 3 seconds
update();
setInterval(update, 3000);
</script>
</body>
</html>
"""


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    """Current agent status including which layer is active."""
    try:
        # Try to get status from webhook server if running
        import urllib.request
        response = urllib.request.urlopen('http://localhost:8000/status', timeout=2)
        data = json.loads(response.read())
        return data
    except Exception:
        return {
            "worker_running": False,
            "queue_depth": 0,
            "running": 0,
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "current_task": None,
            "current_layer": 0,
        }


@app.get("/api/tasks")
async def api_tasks():
    """All tasks from webhook server."""
    try:
        import urllib.request
        response = urllib.request.urlopen('http://localhost:8000/tasks', timeout=2)
        return json.loads(response.read())
    except Exception:
        return {"tasks": [], "total": 0}


@app.get("/api/memory")
async def api_memory():
    """Memory store statistics from Layer 7."""
    try:
        from layer7_memory.memory_store import EpisodicMemoryStore
        memory_path = os.getenv('MEMORY_STORE_PATH', str(ROOT / 'data' / 'memory_store'))
        store = EpisodicMemoryStore(store_path=memory_path)
        return store.get_statistics()
    except Exception as e:
        return {"total": 0, "error": str(e)}


@app.get("/api/health")
async def api_health():
    """System health checks."""
    checks = []

    # Groq API
    groq_key = os.getenv('GROQ_API_KEY', '')
    checks.append({
        "name": "Groq API",
        "ok": bool(groq_key and len(groq_key) > 10),
        "detail": "Connected" if groq_key else "Not configured"
    })

    # GitHub
    github_token = os.getenv('GITHUB_TOKEN', '')
    checks.append({
        "name": "GitHub Token",
        "ok": bool(github_token and len(github_token) > 10),
        "detail": f"@{os.getenv('GITHUB_USERNAME', 'unknown')}" if github_token else "Not set"
    })

    # Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
        checks.append({"name": "Docker", "ok": True, "detail": "Running"})
    except Exception:
        checks.append({"name": "Docker", "ok": False, "warning": True, "detail": "Not running"})

    # Webhook server
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost:8000/health', timeout=1)
        checks.append({"name": "Webhook Server", "ok": True, "detail": "Port 8000"})
    except Exception:
        checks.append({"name": "Webhook Server", "ok": False, "warning": True, "detail": "Not running"})

    # Memory store
    memory_path = ROOT / 'data' / 'memory_store'
    mem_count = len(list(memory_path.glob('*.json'))) if memory_path.exists() else 0
    checks.append({"name": "Memory Store", "ok": True, "detail": f"{mem_count} records"})

    return {"checks": checks, "timestamp": datetime.utcnow().isoformat()}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Autonomous AI Engineer — Dashboard")
    print("="*50)
    print(f"\n  Open in browser: http://localhost:{args.port}")
    print(f"\n  Ctrl+C to stop")
    print("="*50 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
