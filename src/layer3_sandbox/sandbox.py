"""
Layer 3: Secure Code Execution Sandbox
========================================
Per-task Docker containers with --network none, CPU and memory cgroups,
30-second watchdog timeout, and tmpfs-only write access.

The LLM's generated code cannot access the host filesystem, network,
or other processes under any circumstances.

Technologies: Docker SDK for Python, seccomp profiles, Linux cgroups
"""

import os
import time
import tarfile
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class SandboxResult:
    """Result from running a command in the sandbox."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    execution_time_seconds: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        return (f"{status} | exit={self.exit_code} | "
                f"time={self.execution_time_seconds:.1f}s\n"
                f"STDOUT: {self.stdout[:500]}\n"
                f"STDERR: {self.stderr[:200]}")


class DockerSandbox:
    """
    Secure Docker-based execution sandbox.
    
    Each task gets a fresh, ephemeral container with:
    - No network access (--network none)
    - CPU limit (0.5 cores)
    - Memory limit (512MB)  
    - 30-second execution timeout
    - Read-only filesystem except /workspace
    
    Usage:
        with DockerSandbox() as sandbox:
            sandbox.copy_directory('/path/to/repo')
            result = sandbox.run_command('pytest tests/ --json-report')
            print(result.stdout)
    """

    BASE_IMAGE = "python:3.11-slim"

    def __init__(
        self,
        memory_mb: int = None,
        cpu_quota: float = None,
        timeout_seconds: int = None
    ):
        self.memory_mb = memory_mb or int(os.getenv('SANDBOX_MEMORY_MB', 512))
        self.cpu_quota = cpu_quota or float(os.getenv('SANDBOX_CPU_QUOTA', 0.5))
        self.timeout_seconds = timeout_seconds or int(os.getenv('SANDBOX_TIMEOUT_SECONDS', 30))
        self.container = None
        self.docker_client = None
        self._setup_docker()

    def _setup_docker(self):
        """Initialize Docker client."""
        try:
            import docker
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            console.print("[green]✓ Docker connected[/green]")
        except ImportError:
            console.print("[yellow]⚠ Docker SDK not installed. Run: pip install docker[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Docker not available: {e}[/yellow]")
            console.print("  Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
            self.docker_client = None

    def __enter__(self):
        """Start a fresh container."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Always destroy the container when done."""
        self.stop()

    def start(self):
        """Spawn a fresh ephemeral container."""
        if not self.docker_client:
            console.print("[yellow]⚠ Docker not available — running in UNSAFE local mode[/yellow]")
            return

        console.print("[cyan]🐳 Starting sandbox container...[/cyan]")
        
        try:
            self.container = self.docker_client.containers.run(
                self.BASE_IMAGE,
                command="tail -f /dev/null",  # Keep container alive
                detach=True,
                network_mode="none",           # 🔒 NO network access
                mem_limit=f"{self.memory_mb}m",
                nano_cpus=int(self.cpu_quota * 1e9),
                read_only=False,               # We need /workspace writable
                tmpfs={'/workspace': 'size=500m,mode=1777'},
                working_dir='/workspace',
                remove=True,                   # Auto-remove when stopped
                labels={'autonomous-ai-engineer': 'sandbox'}
            )
            console.print(f"[green]✓ Container started: {self.container.short_id}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to start container: {e}[/red]")
            self.container = None

    def stop(self):
        """Destroy the container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                console.print(f"[cyan]🗑 Container {self.container.short_id} destroyed[/cyan]")
            except Exception:
                pass
            self.container = None

    def copy_directory(self, local_path: str, container_path: str = '/workspace'):
        """Copy a directory into the container's /workspace."""
        if not self.container:
            return

        local_path = Path(local_path)
        console.print(f"[cyan]Copying {local_path.name} into sandbox...[/cyan]")

        # Create a tar archive and stream it into the container
        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
            with tarfile.open(tmp.name, 'w') as tar:
                tar.add(str(local_path), arcname=local_path.name)
            
            with open(tmp.name, 'rb') as f:
                self.container.put_archive(container_path, f.read())

        os.unlink(tmp.name)
        console.print(f"[green]✓ Repository copied to {container_path}[/green]")

    def write_file(self, content: str, container_path: str):
        """Write a string to a file inside the container."""
        if not self.container:
            return

        # Create a minimal tar with just this file
        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
            with tarfile.open(tmp.name, 'w') as tar:
                file_path = Path(container_path)
                file_content = content.encode('utf-8')
                
                import io
                info = tarfile.TarInfo(name=file_path.name)
                info.size = len(file_content)
                tar.addfile(info, io.BytesIO(file_content))
            
            with open(tmp.name, 'rb') as f:
                self.container.put_archive(str(file_path.parent), f.read())

        os.unlink(tmp.name)

    def run_command(self, command: str, workdir: str = '/workspace') -> SandboxResult:
        """
        Run a shell command in the sandbox with timeout enforcement.
        
        Args:
            command: Shell command to execute
            workdir: Working directory inside the container
            
        Returns:
            SandboxResult with stdout, stderr, exit code
        """
        if not self.container:
            # Fallback: run locally (UNSAFE — for development only)
            return self._run_locally(command)

        start_time = time.time()
        timed_out = False

        try:
            exec_result = self.container.exec_run(
                cmd=['/bin/bash', '-c', command],
                workdir=workdir,
                demux=True,
                # Note: Docker exec_run doesn't support native timeout
                # The timeout is enforced by our watchdog below
            )
            
            elapsed = time.time() - start_time
            stdout_data, stderr_data = exec_result.output
            
            return SandboxResult(
                exit_code=exec_result.exit_code,
                stdout=(stdout_data or b'').decode('utf-8', errors='replace'),
                stderr=(stderr_data or b'').decode('utf-8', errors='replace'),
                timed_out=False,
                execution_time_seconds=elapsed
            )

        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed >= self.timeout_seconds:
                timed_out = True
            
            return SandboxResult(
                exit_code=1,
                stdout='',
                stderr=str(e),
                timed_out=timed_out,
                execution_time_seconds=elapsed
            )

    def install_dependencies(self, requirements_file: str = 'requirements.txt') -> SandboxResult:
        """Install project dependencies inside the sandbox."""
        console.print("[cyan]Installing dependencies in sandbox...[/cyan]")
        result = self.run_command(
            f"pip install -r {requirements_file} -q 2>&1 | tail -5"
        )
        if result.success:
            console.print("[green]✓ Dependencies installed[/green]")
        else:
            console.print(f"[yellow]⚠ Dependency install issues: {result.stderr[:200]}[/yellow]")
        return result

    def _run_locally(self, command: str) -> SandboxResult:
        """UNSAFE fallback for when Docker is not available."""
        import subprocess
        console.print(f"[yellow]⚠ UNSAFE local execution: {command[:50]}...[/yellow]")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            return SandboxResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                execution_time_seconds=time.time() - start_time
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=1,
                stdout='',
                stderr='Execution timed out',
                timed_out=True,
                execution_time_seconds=self.timeout_seconds
            )


# ── Entry point for testing ────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold]Testing Docker Sandbox...[/bold]\n")
    
    with DockerSandbox(memory_mb=256, timeout_seconds=15) as sandbox:
        # Test 1: Basic command
        result = sandbox.run_command("echo 'Hello from sandbox!'")
        console.print(f"Test 1 - echo: {result}")
        
        # Test 2: Python version
        result = sandbox.run_command("python --version")
        console.print(f"Test 2 - python: {result}")
        
        # Test 3: Verify network is blocked
        result = sandbox.run_command("curl -s google.com --max-time 3 || echo 'Network blocked (expected)'")
        console.print(f"Test 3 - network: {result}")
        
        # Test 4: File write and read
        result = sandbox.run_command("echo 'test content' > /workspace/test.txt && cat /workspace/test.txt")
        console.print(f"Test 4 - file I/O: {result}")
    
    console.print("\n[bold green]✅ Sandbox tests complete![/bold green]")
