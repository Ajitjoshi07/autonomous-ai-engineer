"""
Webhook Setup & ngrok Launcher
================================
Sets up ngrok tunnel and configures the GitHub webhook automatically.

What this does:
  1. Starts the webhook server (FastAPI on port 8000)
  2. Starts ngrok to expose it to the internet
  3. Reads the public ngrok URL
  4. Automatically sets the webhook URL on your GitHub repo
  5. Prints the trigger label to add to issues

Run: python scripts\setup_webhook.py --repo Ajitjoshi07/practice-repo
"""

import os
import sys
import time
import subprocess
import threading
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(ROOT / 'config' / '.env')

from rich.console import Console
from rich.panel import Panel

console = Console()

GITHUB_TOKEN   = os.getenv('GITHUB_TOKEN', '')
TRIGGER_LABEL  = os.getenv('AGENT_TRIGGER_LABEL', 'agent-fix')
WEBHOOK_SECRET = os.getenv('GITHUB_WEBHOOK_SECRET', 'autonomous-ai-engineer-secret')


def check_ngrok():
    """Check if ngrok is installed."""
    try:
        result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✓ ngrok found: {result.stdout.strip()}[/green]")
            return True
    except FileNotFoundError:
        pass
    console.print("[yellow]⚠ ngrok not found[/yellow]")
    console.print("  Download free from: [cyan]https://ngrok.com/download[/cyan]")
    console.print("  After downloading, add ngrok.exe to your PATH or project folder")
    return False


def start_server_thread():
    """Start the webhook server in a background thread."""
    def run():
        subprocess.run(
            [sys.executable, str(ROOT / 'src' / 'webhook_server.py')],
            cwd=str(ROOT)
        )
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    console.print("[green]✓ Webhook server starting on port 8000...[/green]")
    time.sleep(3)
    return thread


def get_ngrok_url():
    """Start ngrok and get the public URL."""
    console.print("[cyan]Starting ngrok tunnel...[/cyan]")

    ngrok_proc = subprocess.Popen(
        ['ngrok', 'http', '8000', '--log=stdout'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for ngrok to start and get URL from its API
    time.sleep(4)
    try:
        import urllib.request
        import json
        response = urllib.request.urlopen('http://localhost:4040/api/tunnels', timeout=5)
        data = json.loads(response.read())
        tunnels = data.get('tunnels', [])
        for tunnel in tunnels:
            if tunnel.get('proto') == 'https':
                url = tunnel['public_url']
                console.print(f"[green]✓ ngrok URL: {url}[/green]")
                return url, ngrok_proc
    except Exception as e:
        console.print(f"[yellow]Could not get ngrok URL automatically: {e}[/yellow]")

    console.print("[yellow]Please copy the ngrok HTTPS URL manually from the ngrok window[/yellow]")
    url = input("Paste your ngrok HTTPS URL here: ").strip()
    return url, ngrok_proc


def setup_github_webhook(repo_name: str, webhook_url: str):
    """Configure the webhook on the GitHub repository."""
    console.print(f"[cyan]Setting up GitHub webhook for {repo_name}...[/cyan]")

    try:
        from github import Github, Auth, GithubException
        gh = Github(auth=Auth.Token(GITHUB_TOKEN))
        repo = gh.get_repo(repo_name)

        webhook_endpoint = f"{webhook_url}/webhook"

        # Check if webhook already exists
        existing = list(repo.get_hooks())
        for hook in existing:
            if 'webhook_server' in str(hook.config.get('url', '')) or \
               'ngrok' in str(hook.config.get('url', '')):
                hook.delete()
                console.print("[yellow]Deleted old webhook[/yellow]")

        # Create new webhook
        repo.create_hook(
            name="web",
            config={
                "url": webhook_endpoint,
                "content_type": "json",
                "secret": WEBHOOK_SECRET,
                "insecure_ssl": "0"
            },
            events=["issues", "issue_comment"],
            active=True
        )
        console.print(f"[green]✓ Webhook created: {webhook_endpoint}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Failed to set webhook: {e}[/red]")
        console.print("\nSet it manually:")
        console.print(f"  1. Go to: https://github.com/{repo_name}/settings/hooks")
        console.print(f"  2. Click 'Add webhook'")
        console.print(f"  3. Payload URL: {webhook_url}/webhook")
        console.print(f"  4. Content type: application/json")
        console.print(f"  5. Secret: {WEBHOOK_SECRET}")
        console.print(f"  6. Events: 'Issues' only")
        return False


def ensure_trigger_label(repo_name: str):
    """Make sure the trigger label exists on the repo."""
    try:
        from github import Github, Auth, GithubException
        gh = Github(auth=Auth.Token(GITHUB_TOKEN))
        repo = gh.get_repo(repo_name)

        existing_labels = [l.name for l in repo.get_labels()]
        if TRIGGER_LABEL not in existing_labels:
            repo.create_label(
                name=TRIGGER_LABEL,
                color="0075ca",
                description="Triggers the Autonomous AI Engineer to fix this issue"
            )
            console.print(f"[green]✓ Label '{TRIGGER_LABEL}' created[/green]")
        else:
            console.print(f"[green]✓ Label '{TRIGGER_LABEL}' already exists[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not create label: {e}[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Setup webhook for Autonomous AI Engineer")
    parser.add_argument('--repo', type=str, default='Ajitjoshi07/practice-repo',
                        help='GitHub repo (owner/repo)')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-ngrok', action='store_true',
                        help='Skip ngrok (if you have your own public URL)')
    parser.add_argument('--url', type=str, help='Your public URL (if not using ngrok)')
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold white]Webhook Setup — Autonomous AI Engineer[/bold white]\n"
        f"[dim]Repo: {args.repo}[/dim]",
        border_style="cyan"
    ))

    # Step 1: Add WEBHOOK_SECRET to .env if not set
    env_file = ROOT / 'config' / '.env'
    env_content = env_file.read_text(encoding='utf-8')
    if 'GITHUB_WEBHOOK_SECRET' not in env_content:
        with open(env_file, 'a') as f:
            f.write(f"\n# Webhook\nGITHUB_WEBHOOK_SECRET={WEBHOOK_SECRET}\n")
            f.write(f"AGENT_TRIGGER_LABEL=agent-fix\n")
            f.write(f"MAX_QUEUE_SIZE=10\n")
        console.print("[green]✓ Webhook config added to config/.env[/green]")

    # Step 2: Ensure label exists
    ensure_trigger_label(args.repo)

    # Step 3: Get public URL
    if args.url:
        public_url = args.url.rstrip('/')
        console.print(f"[green]✓ Using provided URL: {public_url}[/green]")
        ngrok_proc = None
    elif not args.no_ngrok and check_ngrok():
        # Start server then ngrok
        start_server_thread()
        public_url, ngrok_proc = get_ngrok_url()
    else:
        console.print("\n[yellow]Running without ngrok.[/yellow]")
        console.print("Start the server manually: python src\\webhook_server.py")
        public_url = input("Enter your public URL: ").strip().rstrip('/')
        ngrok_proc = None

    # Step 4: Set GitHub webhook
    if public_url:
        setup_github_webhook(args.repo, public_url)

    # Step 5: Print final instructions
    console.print(Panel(
        f"[bold green]✅ Webhook is LIVE![/bold green]\n\n"
        f"Webhook URL:   {public_url}/webhook\n"
        f"Health check:  {public_url}/health\n"
        f"Task status:   {public_url}/status\n\n"
        f"[bold]To trigger the agent:[/bold]\n\n"
        f"  1. Go to: https://github.com/{args.repo}/issues\n"
        f"  2. Open any issue\n"
        f"  3. Add the label: [bold cyan]{TRIGGER_LABEL}[/bold cyan]\n"
        f"  4. Watch the agent fix it and open a PR automatically!\n\n"
        f"[dim]The agent will post a comment on the issue when it starts,[/dim]\n"
        f"[dim]and open a Pull Request when it's done.[/dim]",
        title="🚀 Ready",
        border_style="green"
    ))

    # Keep alive if server is running
    if ngrok_proc:
        console.print("\n[dim]Server running. Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
            ngrok_proc.terminate()


if __name__ == "__main__":
    main()
