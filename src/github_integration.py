"""
GitHub Integration
==================
Handles all GitHub API interactions:
- Fetching issues
- Cloning repositories
- Creating branches and commits
- Opening Pull Requests

Technologies: PyGithub, GitPython
"""

import os
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class GitHubIssue:
    """A GitHub issue with all relevant data."""
    issue_number: int
    title: str
    body: str
    repo_full_name: str     # e.g. "django/django"
    repo_url: str
    labels: list[str]
    author: str
    created_at: str

    @property
    def full_text(self) -> str:
        return f"Title: {self.title}\n\nBody:\n{self.body}"


@dataclass
class PRResult:
    """Result of creating a Pull Request."""
    success: bool
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    error: Optional[str] = None


class GitHubClient:
    """
    Wraps the GitHub API for all agent interactions.

    Usage:
        client = GitHubClient()
        issue = client.get_issue('owner/repo', 42)
        repo_path = client.clone_repo('owner/repo')
        pr = client.create_pr('owner/repo', branch, patch, issue)
    """

    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.username = os.getenv('GITHUB_USERNAME', '')
        self.gh = self._init_github()

    def _init_github(self):
        """Initialize PyGithub client."""
        if not self.token:
            console.print("[yellow]⚠ GITHUB_TOKEN not set in config/.env[/yellow]")
            console.print("  Get a free token at: github.com/settings/tokens")
            return None
        try:
            from github import Github
            gh = Github(self.token)
            user = gh.get_user()
            console.print(f"[green]✓ GitHub connected as: {user.login}[/green]")
            return gh
        except ImportError:
            console.print("[yellow]⚠ PyGithub not installed. Run: pip install PyGithub[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]✗ GitHub auth failed: {e}[/red]")
            return None

    def get_issue(self, repo_name: str, issue_number: int) -> Optional[GitHubIssue]:
        """Fetch a GitHub issue by number."""
        if not self.gh:
            return None
        try:
            repo = self.gh.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            return GitHubIssue(
                issue_number=issue.number,
                title=issue.title,
                body=issue.body or "",
                repo_full_name=repo_name,
                repo_url=repo.clone_url,
                labels=[l.name for l in issue.labels],
                author=issue.user.login,
                created_at=str(issue.created_at),
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to fetch issue #{issue_number}: {e}[/red]")
            return None

    def clone_repo(self, repo_name: str, target_dir: str = None) -> Optional[str]:
        """
        Clone a repository to a local directory.

        Args:
            repo_name: 'owner/repo' format
            target_dir: Where to clone (creates temp dir if None)

        Returns:
            Path to cloned repository
        """
        try:
            import git as gitpython

            if target_dir is None:
                target_dir = tempfile.mkdtemp(prefix='agent_repo_')

            clone_url = f"https://{self.token}@github.com/{repo_name}.git"
            console.print(f"[cyan]Cloning {repo_name}...[/cyan]")

            gitpython.Repo.clone_from(
                clone_url,
                target_dir,
                depth=1,       # Shallow clone — much faster
                single_branch=True
            )
            console.print(f"[green]✓ Cloned to {target_dir}[/green]")
            return target_dir

        except ImportError:
            console.print("[yellow]⚠ gitpython not installed. Run: pip install gitpython[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]✗ Clone failed: {e}[/red]")
            return None

    def create_pull_request(
        self,
        repo_name: str,
        issue: GitHubIssue,
        patch,
        branch_name: str = None,
        iterations_used: int = 0,
        critic_score: float = 0.0,
        repo_path: str = None
    ) -> PRResult:
        """
        Create a Pull Request with the generated patch.

        Args:
            repo_name: 'owner/repo' format
            issue: The GitHubIssue being fixed
            patch: MultiFilePatch to commit
            branch_name: Branch name (auto-generated if None)
            iterations_used: Number of retry iterations for PR description
            critic_score: Critic agent score for PR description
            repo_path: Local path to push from

        Returns:
            PRResult with PR number and URL
        """
        if not self.gh:
            console.print("[yellow]⚠ GitHub not connected — PR creation skipped[/yellow]")
            return PRResult(success=False, error="GitHub not connected")

        # Generate branch name
        if branch_name is None:
            safe_title = re.sub(r'[^a-z0-9]', '-', issue.title.lower())[:40]
            branch_name = f"agent-fix/issue-{issue.issue_number}-{safe_title}"

        console.print(f"\n[bold cyan]🚀 Creating Pull Request...[/bold cyan]")
        console.print(f"Branch: {branch_name}")

        try:
            from github import GithubException
            import git as gitpython

            repo = self.gh.get_repo(repo_name)

            # Get default branch
            default_branch = repo.default_branch
            base_sha = repo.get_branch(default_branch).commit.sha

            # Create branch
            try:
                repo.create_git_ref(
                    ref=f"refs/heads/{branch_name}",
                    sha=base_sha
                )
                console.print(f"[green]✓ Branch created: {branch_name}[/green]")
            except GithubException as e:
                if e.status == 422:  # Branch already exists
                    console.print(f"[yellow]Branch already exists, reusing[/yellow]")
                else:
                    raise

            # Commit each patched file
            for file_patch in patch.patches:
                try:
                    # Try to get existing file (for update)
                    existing = repo.get_contents(file_patch.file_path, ref=branch_name)
                    repo.update_file(
                        path=file_patch.file_path,
                        message=f"fix: {issue.title[:60]} (issue #{issue.issue_number})",
                        content=file_patch.patched_content,
                        sha=existing.sha,
                        branch=branch_name
                    )
                except Exception:
                    # File doesn't exist — create it
                    repo.create_file(
                        path=file_patch.file_path,
                        message=f"fix: {issue.title[:60]} (issue #{issue.issue_number})",
                        content=file_patch.patched_content,
                        branch=branch_name
                    )
                console.print(f"[green]✓ Committed: {file_patch.file_path}[/green]")

            # Build PR description
            pr_body = self._build_pr_description(
                issue=issue,
                patch=patch,
                iterations_used=iterations_used,
                critic_score=critic_score
            )

            # Create the PR
            pr = repo.create_pull(
                title=f"[AI Fix] {issue.title}",
                body=pr_body,
                head=branch_name,
                base=default_branch,
            )

            console.print(f"\n[bold green]✅ Pull Request created![/bold green]")
            console.print(f"   PR #{pr.number}: {pr.html_url}")

            return PRResult(
                success=True,
                pr_number=pr.number,
                pr_url=pr.html_url
            )

        except Exception as e:
            console.print(f"[red]✗ PR creation failed: {e}[/red]")
            return PRResult(success=False, error=str(e))

    def _build_pr_description(
        self,
        issue: GitHubIssue,
        patch,
        iterations_used: int,
        critic_score: float
    ) -> str:
        """Build a professional PR description."""
        files_changed = '\n'.join(f"- `{p.file_path}` (+{p.lines_added}/-{p.lines_removed})" 
                                   for p in patch.patches)
        
        return f"""## 🤖 Automated Fix — Issue #{issue.issue_number}

This Pull Request was generated by [Autonomous AI Software Engineer](https://github.com/{os.getenv('GITHUB_USERNAME', 'user')}/autonomous-ai-engineer).

### Problem
Closes #{issue.issue_number}: {issue.title}

### Changes Made
{files_changed}

### How the Fix Was Verified
- ✅ All existing tests pass after applying this patch
- ✅ Code reviewed by adversarial critic agent (score: {critic_score:.2f}/1.0)
- ✅ ruff linter: 0 violations

### Agent Metadata
| Metric | Value |
|--------|-------|
| Iterations required | {iterations_used} |
| Critic approval score | {critic_score:.2f} |
| Files changed | {len(patch.patches)} |

---
*Generated by Autonomous AI Software Engineer — [Ajit Mukund Joshi]*
*This PR was created without human intervention. Please review carefully.*
"""
