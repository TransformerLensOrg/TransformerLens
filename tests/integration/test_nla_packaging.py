"""Built-wheel NLA resources without Hub or model downloads."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

_REQUIRED = (
    "transformer_lens/tools/analysis/nla_registry.json",
    "transformer_lens/tools/analysis/_nla_artifacts.py",
    "transformer_lens/tools/analysis/_nla_contracts.py",
    "transformer_lens/tools/analysis/_nla_target.py",
    "transformer_lens/tools/analysis/_nla_verbalizer.py",
    "transformer_lens/tools/analysis/_nla_reconstructor.py",
    "transformer_lens/tools/analysis/_nla.py",
)


def test_built_wheel_nla_resources(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "dist"
    build_environment = os.environ.copy()
    build_environment["UV_CACHE_DIR"] = str(tmp_path / "fresh-uv-cache")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=repository,
        env=build_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1
    extracted = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in _REQUIRED:
            assert name in wheel.namelist(), f"Missing wheel resource: {name}"
        registry = json.loads(wheel.read(_REQUIRED[0]))
        assert registry["schema_version"] == 1
        assert isinstance(registry["models"], dict)
        wheel.extractall(extracted)
    script = r"""
import pathlib
import socket
import sys

root = pathlib.Path(sys.argv[1])

def forbid_network(*args, **kwargs):
    raise AssertionError("Wheel resource imports must not initiate network access")

socket.socket.connect = forbid_network
socket.create_connection = forbid_network
sys.path.insert(0, str(root))
import transformer_lens
import transformer_lens.tools.analysis
assert root in pathlib.Path(transformer_lens.__file__).parents
assert root in pathlib.Path(transformer_lens.tools.analysis.__file__).parents
from transformer_lens.tools.analysis._nla_artifacts import load_registry
from transformer_lens.tools.analysis._nla_contracts import NLASamplingConfig
assert isinstance(load_registry(), dict)
assert NLASamplingConfig().max_new_tokens == 512
from transformer_lens.tools.analysis import _nla_artifacts
assert root in pathlib.Path(_nla_artifacts.__file__).parents
print("Isolated wheel NLA resource import passed")
"""
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(extracted)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Isolated wheel NLA resource import passed" in result.stdout
