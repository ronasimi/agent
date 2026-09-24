from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_script_exists_and_is_executable():
    script = ROOT / "scripts" / "bootstrap_venv.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111
    text = script.read_text(encoding="utf-8")
    assert "python -m venv" not in text  # interpreter is configurable
    assert ' -m venv "$VENV"' in text
    assert ' -m pip install -e "$ROOT"' in text


def test_model_role_benchmark_bootstraps_repo_venv_before_optional_imports():
    text = (ROOT / "diagnostics" / "benchmarks" / "benchmark_model_roles.py").read_text(encoding="utf-8")
    assert 'VENV_PYTHON = ROOT / ".venv" / "bin" / "python"' in text
    assert "_maybe_reexec_in_repo_venv()" in text
    assert text.index("_maybe_reexec_in_repo_venv()") < text.index("import yaml")
    assert "bootstrap_venv.sh" in text
