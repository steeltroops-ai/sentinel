#!/usr/bin/env python3
"""
KIVE Project Task Runner
Usage: python run.py <task>
Example: python run.py test, python run.py train
"""

import argparse
import subprocess
import sys
from pathlib import Path


class TaskRunner:
    """Task runner for KIVE project - cross-platform alternative to Makefile"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        
    def run_command(self, cmd: str, shell: bool = True):
        """Run a shell command and handle errors"""
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=shell, cwd=self.root)
        if result.returncode != 0:
            print(f"Error: Command failed with code {result.returncode}")
            sys.exit(result.returncode)
    
    # ========================================================================
    # Setup & Installation
    # ========================================================================
    
    def install(self):
        """Install all dependencies with uv"""
        self.run_command("uv sync")
    
    def install_dev(self):
        """Install with dev dependencies"""
        self.run_command("uv sync --all-extras")
    

    # ========================================================================
    # Documentation & Visualization
    # ========================================================================
    
    def update_docs(self):
        """Copy artifacts to docs and update README"""
        self.run_command("uv run python scripts/update_docs.py")
    
    def visualize_mlflow(self):
        """Generate MLflow comparison plots"""
        self.run_command("uv run python scripts/visualize_mlflow.py")
    
    def mlflow_ui(self):
        """Start MLflow UI server"""
        print("Starting MLflow UI at http://localhost:5000")
        self.run_command("uv run mlflow ui")
    
    # ========================================================================
    # Testing
    # ========================================================================
    
    def test(self):
        """Run all tests"""
        self.run_command("uv run pytest tests/ -v")
    
    def test_cov(self):
        """Run tests with coverage report"""
        self.run_command("uv run pytest tests/ -v --cov --cov-report=html --cov-report=term")
    
    def test_fast(self):
        """Run tests in parallel (fast)"""
        self.run_command("uv run pytest tests/ -v -n auto")
    
    def test_integration(self):
        """Run integration tests only"""
        self.run_command("uv run pytest tests/test_integration.py -v")
    
    def test_env(self):
        """Run environment tests only"""
        self.run_command("uv run pytest tests/test_env.py -v")
    
    # ========================================================================
    # Code Quality
    # ========================================================================
    
    def lint(self):
        """Run ruff linter"""
        self.run_command("uv run ruff check .")
    
    def format(self):
        """Format code with black"""
        self.run_command("uv run black .")
    
    def check(self):
        """Run lint + format check"""
        self.lint()
        self.run_command("uv run black --check .")
    
    # ========================================================================
    # Data Generation
    # ========================================================================
    
    def data(self):
        """Generate synthetic profiles (5000)"""
        self.run_command(
            "uv run python data/synthetic_generator.py --n 5000 --fraud-ratio 0.4 "
            "--output data/synthetic_profiles.json"
        )
    
    def data_small(self):
        """Generate small dataset (500)"""
        self.run_command(
            "uv run python data/synthetic_generator.py --n 500 --fraud-ratio 0.4 "
            "--output data/synthetic_profiles.json"
        )
    
    def data_large(self):
        """Generate large dataset (10000)"""
        self.run_command(
            "uv run python data/synthetic_generator.py --n 10000 --fraud-ratio 0.4 "
            "--output data/synthetic_profiles.json"
        )
    
    def data_verbose(self):
        """Generate data with verbose output"""
        self.run_command(
            "uv run python data/synthetic_generator.py --n 5000 --fraud-ratio 0.4 "
            "--verbose --output data/synthetic_profiles.json"
        )
    
    def validate_data(self):
        """Validate signal distributions"""
        self.run_command("uv run python data/validate_distribution.py")
    
    def export_distributions(self):
        """Export signal distributions"""
        self.run_command("uv run python data/export_signal_distributions.py")
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train(self):
        """Train RL agent (3000 episodes) with MLflow tracking"""
        self.run_command(
            "uv run python services/orchestrator/train.py --n-episodes 3000 "
            "--run-name kive_ppo_standard --output-dir artifacts/training"
        )
    
    def train_fast(self):
        """Quick training (1000 episodes) - no MLflow"""
        self.run_command(
            "uv run python services/orchestrator/train.py --n-episodes 1000 "
            "--run-name kive_ppo_fast --output-dir artifacts/training --no-mlflow"
        )
    
    def train_full(self):
        """Full training (10000 episodes) with MLflow tracking"""
        self.run_command(
            "uv run python services/orchestrator/train.py --n-episodes 10000 "
            "--run-name kive_ppo_full --output-dir artifacts/training"
        )
    
    # ========================================================================
    # Docker & Services
    # ========================================================================
    
    def docker_up(self):
        """Start all services with Docker Compose"""
        self.run_command("docker-compose up -d --build")
    
    def docker_down(self):
        """Stop all services"""
        self.run_command("docker-compose down")
    
    def docker_build(self):
        """Rebuild Docker images"""
        self.run_command("docker-compose build")
    
    def docker_logs(self):
        """View service logs"""
        self.run_command("docker-compose logs -f")
    
    def docker_restart(self):
        """Restart all services"""
        self.run_command("docker-compose restart")
    
    def health(self):
        """Check health of all services"""
        import httpx
        services = {
            "TAV": "http://localhost:8001/health",
            "SVP": "http://localhost:8002/health",
            "FMD": "http://localhost:8003/health",
            "MDC": "http://localhost:8004/health",
            "TSI": "http://localhost:8005/health",
            "BES": "http://localhost:8006/health",
            "LQA": "http://localhost:8007/health",
            "CCS": "http://localhost:8008/health",
            "RSL": "http://localhost:8009/health",
        }
        
        print("Checking service health...")
        for name, url in services.items():
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    print(f"✓ {name} service: OK")
                else:
                    print(f"✗ {name} service: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ {name} service: DOWN ({e})")
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def clean(self):
        """Clean artifacts and cache"""
        import shutil
        
        dirs_to_remove = [
            ".pytest_cache",
            ".ruff_cache",
            "htmlcov",
            ".coverage",
            "artifacts/training",
        ]
        
        for dir_path in dirs_to_remove:
            path = self.root / dir_path
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed: {dir_path}")
        
        # Remove __pycache__ directories
        for pycache in self.root.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)
        
        # Remove .pyc files
        for pyc in self.root.rglob("*.pyc"):
            pyc.unlink()
        
        print("Clean complete!")
    
    def clean_all(self):
        """Deep clean (including data)"""
        self.clean()
        
        data_file = self.root / "data" / "synthetic_profiles.json"
        if data_file.exists():
            data_file.unlink()
            print("Removed: data/synthetic_profiles.json")
        
        venv = self.root / ".venv"
        if venv.exists():
            import shutil
            shutil.rmtree(venv, ignore_errors=True)
            print("Removed: .venv")
        
        print("Deep clean complete!")
    
    def notebook(self):
        """Start Jupyter notebook server"""
        self.run_command("uv run jupyter notebook")
    
    # ========================================================================
    # Submission
    # ========================================================================
    
    def submit_check(self):
        """Check submission readiness"""
        print("Checking submission readiness...")
        print("")
        
        checks = []
        
        # Check tests
        print("1. Checking tests...")
        try:
            subprocess.run(
                "uv run pytest tests/ -v --tb=no -q",
                shell=True,
                check=True,
                cwd=self.root
            )
            checks.append(("Tests", True))
        except subprocess.CalledProcessError:
            checks.append(("Tests", False))
        
        # Check memo
        print("\n2. Checking memo...")
        memo_exists = (self.root / "memo.md").exists()
        checks.append(("memo.md", memo_exists))
        
        # Check training artifacts
        print("\n3. Checking training artifacts...")
        report_exists = (self.root / "artifacts" / "training" / "convergence_report.json").exists()
        checks.append(("Training complete", report_exists))
        
        # Check multi-modal doc
        print("\n4. Checking multi-modal doc...")
        multimodal_exists = (self.root / "docs" / "multimodal_live_evaluator.md").exists()
        checks.append(("Multi-modal doc", multimodal_exists))
        
        # Summary
        print("\n" + "="*50)
        print("Submission Checklist:")
        print("="*50)
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {name}")
        
        all_passed = all(passed for _, passed in checks)
        if all_passed:
            print("\n✓ All checks passed! Ready to submit.")
        else:
            print("\n✗ Some checks failed. Fix issues before submitting.")
            sys.exit(1)
    
    def submit_package(self):
        """Package for submission"""
        print("Packaging for submission...")
        print("1. Converting memo to PDF...")
        print("2. Creating submission archive...")
        print("Not implemented - manual submission required")


def main():
    runner = TaskRunner()
    
    # Build command map
    commands = {
        # Setup
        "install": runner.install,
        "install-dev": runner.install_dev,
        # Testing
        "test": runner.test,
        "test-cov": runner.test_cov,
        "test-fast": runner.test_fast,
        "test-integration": runner.test_integration,
        "test-env": runner.test_env,
        # Code quality
        "lint": runner.lint,
        "format": runner.format,
        "check": runner.check,
        # Data
        "data": runner.data,
        "data-small": runner.data_small,
        "data-large": runner.data_large,
        "data-verbose": runner.data_verbose,
        "validate-data": runner.validate_data,
        "export-distributions": runner.export_distributions,
        # Training
        "train": runner.train,
        "train-fast": runner.train_fast,
        "train-full": runner.train_full,
        # Documentation
        "update-docs": runner.update_docs,
        "visualize-mlflow": runner.visualize_mlflow,
        "mlflow-ui": runner.mlflow_ui,
        # Docker
        "docker-up": runner.docker_up,
        "docker-down": runner.docker_down,
        "docker-build": runner.docker_build,
        "docker-logs": runner.docker_logs,
        "docker-restart": runner.docker_restart,
        "health": runner.health,
        # Utilities
        "clean": runner.clean,
        "clean-all": runner.clean_all,
        "notebook": runner.notebook,
        # Submission
        "submit-check": runner.submit_check,
        "submit-package": runner.submit_package,
    }
    
    parser = argparse.ArgumentParser(
        description="KIVE Project Task Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tasks:
  Setup:        install, install-dev
  Testing:      test, test-cov, test-fast, test-integration, test-env
  Code Quality: lint, format, check
  Data:         data, data-small, data-large, data-verbose, validate-data, export-distributions
  Training:     train, train-fast, train-full
  Docs:         update-docs, visualize-mlflow, mlflow-ui
  Docker:       docker-up, docker-down, docker-build, docker-logs, health
  Utilities:    clean, clean-all, notebook
  Submission:   submit-check, submit-package

Examples:
  python run.py test
  python run.py train
  python run.py update-docs
  python run.py mlflow-ui
  python run.py docker-up
        """
    )
    parser.add_argument("task", choices=commands.keys(), help="Task to run")
    
    args = parser.parse_args()
    
    # Run the task
    commands[args.task]()


if __name__ == "__main__":
    main()
