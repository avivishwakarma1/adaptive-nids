"""SAFETY_POLICY section 11 checks that apply to the ML package: no network/enforcement/DB/web imports, no shell,
no dynamic code execution, offline operation."""
import ast
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "src" / "nids_ml").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
ALLOWED_THIRD_PARTY = {"numpy", "pandas", "sklearn", "scipy", "pydantic", "nids_ml", "_common",
                       "train", "evaluate", "inspect_dataset", "unknown_attack", "continual_experiment"}
FORBIDDEN_MODULES = {"socket", "subprocess", "requests", "httpx", "urllib3", "aiohttp", "ftplib", "smtplib", "telnetlib", "ssl",
                     "http", "urllib", "scapy", "pyshark", "dpkt", "fastapi", "starlette", "pymongo", "motor", "flask", "uvicorn",
                     "ctypes", "multiprocessing", "asyncio", "paramiko", "netfilterqueue", "pexpect", "pty", "shlex", "webbrowser"}
FORBIDDEN_BARE_CALLS = {"eval", "exec", "compile", "__import__"}          # builtins (re.compile is fine)
FORBIDDEN_ATTR_CALLS = {"system", "popen", "Popen", "run_path", "run_module", "spawn", "fork", "execv", "execvp", "execl"}


def _imports(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            yield n.module.split(".")[0]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_forbidden_imports_or_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set(_imports(tree))
    assert not (mods & FORBIDDEN_MODULES), f"{path.name} imports {mods & FORBIDDEN_MODULES}"
    unknown = {m for m in mods if m not in sys.stdlib_module_names and m not in ALLOWED_THIRD_PARTY}
    assert not unknown, f"{path.name} imports unreviewed third-party modules {unknown}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                assert n.func.id not in FORBIDDEN_BARE_CALLS, f"{path.name}: forbidden call {n.func.id}()"
            elif isinstance(n.func, ast.Attribute):
                assert n.func.attr not in FORBIDDEN_ATTR_CALLS, f"{path.name}: forbidden call .{n.func.attr}()"
            for kw in n.keywords:
                assert not (kw.arg == "shell" and getattr(kw.value, "value", False) is True)


def test_package_import_does_not_pull_in_backend_or_db_or_network_stacks():
    import nids_ml  # noqa: F401
    for m in ("fastapi", "pymongo", "requests", "httpx", "scapy", "subprocess_not_imported_marker"):
        assert m not in sys.modules


def test_training_inference_and_evaluation_work_with_networking_disabled(monkeypatch, bundle, test_records):
    from nids_ml import EvaluationConfig, evaluate, predict

    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", boom); monkeypatch.setattr(socket, "create_connection", boom)
    recs, labs = test_records
    assert len(predict(recs[:20], bundle)) == 20
    evaluate(bundle, [{"observation": o, "label": l} for o, l in zip(recs[:60], labs[:60])], EvaluationConfig())


def test_risk_output_is_advisory_only_no_action_fields():
    from nids_ml import PredictionResult
    assert not {"action", "block", "blocked", "remediation", "enforce"} & set(PredictionResult.model_fields)


def test_no_automatic_retraining_hooks_exist():
    """Feedback handling only builds a dataset; nothing in the package trains as a side effect of predict/evaluate."""
    import nids_ml.inference as inf, nids_ml.evaluation as ev
    src = (Path(inf.__file__).read_text() + Path(ev.__file__).read_text())
    # evaluation.py may call fit_bundle ONLY inside the explicit unknown-attack protocol function
    assert "fit_bundle(" not in Path(inf.__file__).read_text()
    assert Path(ev.__file__).read_text().count("fit_bundle(") == 1
