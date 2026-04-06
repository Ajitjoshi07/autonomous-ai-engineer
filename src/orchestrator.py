"""
Main Orchestrator: Autonomous AI Software Engineer
===================================================
Connects all 7 layers into a single end-to-end pipeline.

Data flow:
  GitHub Issue
    → [L1] Index repo + retrieve relevant chunks
    → [L7] Query memory for similar past issues
    → [L2] Tree-of-Thought planning
    → [L3] Spawn Docker sandbox
    → [L4] Generate code patch
    → [L5] Run tests → retry loop (up to 8x)
    → [L6] Critic review
    → GitHub PR ✅
    → [L7] Store memory

Usage:
    python src/orchestrator.py --repo owner/repo --issue 42
    python src/orchestrator.py --local-path /path/to/repo --issue-file issue.txt
"""

import os
import sys
import time
import argparse
import tempfile
from pathlib import Path
from typing import Optional

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


class AutonomousEngineer:
    """
    The complete Autonomous AI Software Engineer pipeline.
    Orchestrates all 7 layers for a single GitHub issue.
    """

    def __init__(self):
        console.print(Panel.fit(
            "[bold white]AUTONOMOUS AI SOFTWARE ENGINEER[/bold white]\n"
            "[dim]by Ajit Mukund Joshi[/dim]",
            border_style="bright_blue"
        ))
        self._init_layers()

    def _init_layers(self):
        """Initialize all 7 layers."""
        console.print("\n[bold]Initializing layers...[/bold]")

        # Layer 1: Codebase Understanding
        from layer1_understanding.engine import CodebaseUnderstandingEngine
        self.l1_understanding = CodebaseUnderstandingEngine()

        # Layer 2: Planning Agent
        from layer2_planning.agent import PlanningAgent
        self.l2_planning = PlanningAgent()

        # Layer 3: Sandbox
        from layer3_sandbox.sandbox import DockerSandbox
        self.DockerSandbox = DockerSandbox  # Class reference, instantiate per task

        # Layer 4: Code Generator
        from layer4_codegen.patch_engine import CodeGenerator
        self.l4_codegen = CodeGenerator()

        # Layer 5: Feedback Loop (instantiated per task with sandbox)
        from layer5_feedback.feedback_loop import FeedbackLoop, TestRunner
        self.FeedbackLoop = FeedbackLoop
        self.TestRunner = TestRunner

        # Layer 6: Critic
        from layer6_critic.critic import CriticAgent
        self.l6_critic = CriticAgent()

        # Layer 7: Memory
        from layer7_memory.memory_store import EpisodicMemoryStore, build_memory_record
        self.l7_memory = EpisodicMemoryStore()
        self.build_memory_record = build_memory_record

        # GitHub Integration
        from github_integration import GitHubClient
        self.github = GitHubClient()

        console.print("[bold green]✅ All layers initialized[/bold green]\n")

    def run(
        self,
        issue_text: str,
        repo_name: str = None,
        repo_path: str = None,
        issue_number: int = 0
    ) -> dict:
        """
        Run the full pipeline on a GitHub issue.

        Args:
            issue_text: The full text of the GitHub issue
            repo_name: 'owner/repo' (for GitHub API + cloning)
            repo_path: Local path to already-cloned repo (skips cloning)
            issue_number: GitHub issue number (for PR creation)

        Returns:
            Result dict with success, pr_url, iterations, cost, etc.
        """
        start_time = time.time()
        result = {
            'success': False,
            'iterations': 0,
            'pr_url': None,
            'pr_number': None,
            'error': None,
            'time_seconds': 0,
        }

        console.print(Panel(
            issue_text[:600] + ("..." if len(issue_text) > 600 else ""),
            title=f"📋 Issue {'#' + str(issue_number) if issue_number else ''}",
            border_style="yellow"
        ))

        try:
            # ── Step 1: Clone repo if needed ─────────────────────────────
            if repo_path is None and repo_name:
                console.print(Rule("[cyan]Step 1: Cloning Repository[/cyan]"))
                repo_path = self.github.clone_repo(repo_name)
                if not repo_path:
                    result['error'] = "Failed to clone repository"
                    return result
            elif repo_path is None:
                result['error'] = "Must provide either repo_name or repo_path"
                return result

            # ── Step 2: Layer 1 — Index Codebase ─────────────────────────
            console.print(Rule("[cyan]Step 2: Indexing Codebase (Layer 1)[/cyan]"))
            self.l1_understanding.index_repository(repo_path)

            # ── Step 3: Retrieve relevant chunks ─────────────────────────
            console.print(Rule("[cyan]Step 3: Semantic Retrieval[/cyan]"))
            retrieved = self.l1_understanding.query(issue_text, top_k=5)
            retrieved_chunks = [
                {
                    'name': chunk.name,
                    'file_path': chunk.file_path,
                    'source_code': chunk.source_code,
                    'start_line': chunk.start_line,
                }
                for chunk, score in retrieved
            ]
            console.print(f"[green]Retrieved {len(retrieved_chunks)} relevant code chunks[/green]")

            # ── Step 4: Layer 7 — Query Memory ───────────────────────────
            console.print(Rule("[cyan]Step 4: Memory Retrieval (Layer 7)[/cyan]"))
            similar_memories = self.l7_memory.retrieve_similar(issue_text, top_k=3)
            memory_context = ""
            if similar_memories:
                memory_context = "Past similar issues:\n\n" + "\n\n".join(
                    m.to_prompt_text() for m in similar_memories
                )

            # ── Step 5: Layer 2 — Planning ────────────────────────────────
            console.print(Rule("[cyan]Step 5: Tree-of-Thought Planning (Layer 2)[/cyan]"))

            # Build call graph context for the most relevant function
            call_graph_context = ""
            if retrieved_chunks:
                top_fn = retrieved_chunks[0]['name']
                callers = self.l1_understanding.get_callers(top_fn)
                if callers:
                    call_graph_context = f"Function '{top_fn}' is called by: {', '.join(callers[:5])}"

            plan = self.l2_planning.create_plan(
                issue_text=issue_text,
                retrieved_chunks=retrieved_chunks,
                call_graph_context=call_graph_context,
                memory_context=memory_context
            )

            if not plan:
                result['error'] = "Planning failed — could not generate repair plan"
                return result

            # ── Step 6: Layer 3 — Spawn Sandbox + Layer 4 — Generate Patch ──
            console.print(Rule("[cyan]Step 6: Sandbox + Code Generation (L3 + L4)[/cyan]"))

            with self.DockerSandbox() as sandbox:
                # Copy repo into sandbox
                sandbox.copy_directory(repo_path)

                # Install dependencies
                sandbox.install_dependencies()

                # Generate initial patch
                patch = self.l4_codegen.generate_patch(
                    plan=plan,
                    retrieved_chunks=retrieved_chunks,
                    repo_path=repo_path,
                    sandbox=sandbox
                )

                if not patch:
                    result['error'] = "Code generation failed — no patch produced"
                    return result

                # Apply the patch to the sandbox
                self.l4_codegen.apply_patch(patch, sandbox=sandbox)

                # ── Step 7: Layer 5 — Test Feedback Loop ─────────────────
                console.print(Rule("[cyan]Step 7: Test Feedback Loop (Layer 5)[/cyan]"))

                feedback_loop = self.FeedbackLoop(
                    sandbox=sandbox,
                    code_generator=self.l4_codegen
                )

                tests_passed, iterations, test_history = feedback_loop.run(
                    plan=plan,
                    repo_path='/workspace'
                )

                result['iterations'] = iterations

                if not tests_passed:
                    console.print("[yellow]⚠ Tests did not fully pass — proceeding to critic anyway[/yellow]")

                # ── Step 8: Layer 6 — Critic Review ──────────────────────
                console.print(Rule("[cyan]Step 8: Critic Review (Layer 6)[/cyan]"))

                # Get the final patch state
                final_patch = self.l4_codegen.patch_history[-1] if self.l4_codegen.patch_history else patch
                critic_feedback = self.l6_critic.review(
                    patch=final_patch,
                    issue_text=issue_text
                )

                if not critic_feedback.approved:
                    console.print(f"[red]Critic rejected patch: {critic_feedback.rejection_reason}[/red]")
                    # In a full implementation, we'd retry with critic feedback
                    # For now, log and continue
                    console.print("[yellow]Proceeding anyway (implement retry loop in Phase 3)[/yellow]")

            # ── Step 9: Create Pull Request ───────────────────────────────
            pr_result = None
            if tests_passed and critic_feedback.approved and repo_name and issue_number:
                console.print(Rule("[cyan]Step 9: Creating Pull Request[/cyan]"))

                from github_integration import GitHubIssue
                issue_obj = GitHubIssue(
                    issue_number=issue_number,
                    title=plan.issue_summary,
                    body=issue_text,
                    repo_full_name=repo_name,
                    repo_url=f"https://github.com/{repo_name}",
                    labels=[],
                    author="unknown",
                    created_at=""
                )

                pr_result = self.github.create_pull_request(
                    repo_name=repo_name,
                    issue=issue_obj,
                    patch=final_patch,
                    iterations_used=iterations,
                    critic_score=critic_feedback.overall_score
                )

                if pr_result.success:
                    result['pr_url'] = pr_result.pr_url
                    result['pr_number'] = pr_result.pr_number

            # ── Step 10: Layer 7 — Store Memory ──────────────────────────
            console.print(Rule("[cyan]Step 10: Storing Memory (Layer 7)[/cyan]"))

            memory_record = self.build_memory_record(
                repository=repo_name or repo_path,
                issue_text=issue_text,
                plan=plan,
                test_results=test_history,
                critic_feedback=critic_feedback,
                success=tests_passed and critic_feedback.approved,
                start_time=start_time,
                pr_url=result.get('pr_url'),
                pr_number=result.get('pr_number')
            )
            self.l7_memory.store(memory_record)

            # ── Final Summary ─────────────────────────────────────────────
            elapsed = time.time() - start_time
            result['success'] = tests_passed and critic_feedback.approved
            result['time_seconds'] = elapsed

            self._print_final_summary(result, elapsed, plan, critic_feedback)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            result['error'] = "Interrupted"
        except Exception as e:
            console.print(f"\n[red]✗ Pipeline error: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            result['error'] = str(e)

        result['time_seconds'] = time.time() - start_time
        return result

    def _print_final_summary(self, result, elapsed, plan, critic_feedback):
        """Print a final summary panel."""
        status = "[bold green]✅ SUCCESS[/bold green]" if result['success'] \
            else "[bold red]❌ INCOMPLETE[/bold red]"

        lines = [
            f"Status: {status}",
            f"Issue: {plan.issue_summary if plan else 'unknown'}",
            f"Iterations: {result['iterations']}",
            f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
            f"Critic Score: {critic_feedback.overall_score:.2f}" if critic_feedback else "",
        ]
        if result.get('pr_url'):
            lines.append(f"PR: {result['pr_url']}")

        console.print(Panel(
            '\n'.join(l for l in lines if l),
            title="Pipeline Complete",
            border_style="green" if result['success'] else "yellow"
        ))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Autonomous AI Software Engineer — fix GitHub issues automatically'
    )
    parser.add_argument('--repo', type=str, help='GitHub repo (owner/repo format)')
    parser.add_argument('--issue', type=int, help='GitHub issue number')
    parser.add_argument('--issue-text', type=str, help='Issue text directly')
    parser.add_argument('--issue-file', type=str, help='Path to file containing issue text')
    parser.add_argument('--local-path', type=str, help='Local repo path (skips cloning)')
    args = parser.parse_args()

    # Get issue text
    issue_text = None
    if args.issue_text:
        issue_text = args.issue_text
    elif args.issue_file:
        issue_text = Path(args.issue_file).read_text()
    elif args.issue and args.repo:
        # Fetch from GitHub
        from github_integration import GitHubClient
        client = GitHubClient()
        issue = client.get_issue(args.repo, args.issue)
        if issue:
            issue_text = issue.full_text
        else:
            console.print("[red]Failed to fetch issue from GitHub[/red]")
            sys.exit(1)
    else:
        # Interactive mode
        console.print("[cyan]Enter the GitHub issue text (press Ctrl+D when done):[/cyan]")
        try:
            issue_text = sys.stdin.read()
        except EOFError:
            pass

    if not issue_text or not issue_text.strip():
        console.print("[red]No issue text provided. Use --issue-text or --issue + --repo[/red]")
        parser.print_help()
        sys.exit(1)

    # Run the pipeline
    engineer = AutonomousEngineer()
    result = engineer.run(
        issue_text=issue_text,
        repo_name=args.repo,
        repo_path=args.local_path,
        issue_number=args.issue or 0
    )

    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
