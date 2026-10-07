"""
Hardware / model provenance recorded in run_config.json (Goal A, Tugas 4): never fails, writes None when missing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate_benchmark as eb  # noqa: E402

HOST_KEYS = {"cpu", "cpu_logical_cores", "ram_bytes", "ram_gb", "os", "os_build", "machine", "python",
             "python_executable", "power_source"}


def test_host_info_has_all_keys():
    info = eb._host_info()
    assert set(info) == HOST_KEYS
    assert info["python"] == sys.version.split()[0]


def test_host_info_survives_missing_commands(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such command")
    monkeypatch.setattr(subprocess, "run", boom)
    info = eb._host_info()
    assert set(info) == HOST_KEYS
    for k in ("cpu_logical_cores", "ram_bytes", "ram_gb", "os", "os_build", "power_source"):
        assert info[k] is None, k


def test_parse_ollama_list():
    text = ("NAME                            ID              SIZE      MODIFIED     \n"
            "gemma4:e2b                      7fbdbf8f5e45    7.2 GB    5 months ago    \n")
    assert eb._parse_ollama_list(text) == {"gemma4:e2b": {"digest": "7fbdbf8f5e45", "size": "7.2 GB"}}
    assert eb._parse_ollama_list("") == {}


def test_model_info_is_none_when_ollama_unreachable(monkeypatch):
    import urllib.request

    def down(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", down)
    monkeypatch.setattr(eb, "_cmd_out", lambda cmd: None)
    assert eb._ollama_model_info(["gemma4:e2b", "llama3.1:8b"]) == {"gemma4:e2b": None, "llama3.1:8b": None}
    assert eb._ollama_version() is None
