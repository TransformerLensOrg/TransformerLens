TL_UV_ACTIVE ?= 0
ACTIVE_FLAG := $(if $(filter 1 true TRUE yes YES on ON,$(TL_UV_ACTIVE)), --active,)
RUN := uv run$(ACTIVE_FLAG)
UV_SYNC := uv sync$(ACTIVE_FLAG)

# Rerun args for flaky tests (httpx timeouts during HF Hub downloads)
# Remove this line when no longer needed
RERUN_ARGS := --reruns 2 --reruns-delay 5

# Parallelism for the coverage run. loadscope keeps a module's tests + its
# module/session fixtures on one worker (models load once per worker, not
# re-scattered). Override to -n 2 (or -n 0) if a runner OOMs.
XDIST_ARGS ?= -n auto --dist loadscope

dep:
	$(UV_SYNC)

format:
	$(RUN) pycln --all . --exclude "__init__.py"
	$(RUN) isort format .
	$(RUN) black .

check-format:
	$(RUN) pycln --check --all . --exclude "__init__.py"
	$(RUN) isort --check-only .
	$(RUN) black --check .

unit-test:
	$(RUN) pytest tests/unit -m "not slow" $(RERUN_ARGS)

integration-test:
	$(RUN) pytest tests/integration -m "not slow" $(RERUN_ARGS)

acceptance-test:
	$(RUN) pytest tests/acceptance -m "not slow" $(RERUN_ARGS)

benchmark-test:
	$(RUN) pytest tests/benchmarks $(RERUN_ARGS)

coverage-report-test:
	$(RUN) pytest -o "addopts=--jaxtyping-packages=transformer_lens,beartype.beartype -W ignore::beartype.roar.BeartypeDecorHintPep585DeprecationWarning" --cov=transformer_lens/ --cov-report=html --cov-branch $(XDIST_ARGS) -m "not slow" tests/integration tests/unit tests/acceptance $(RERUN_ARGS)

docstring-test:
	$(RUN) pytest transformer_lens/ $(RERUN_ARGS)

notebook-test:
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/BERT.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Bridge_Evals_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Exploratory_Analysis_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Main_Demo.ipynb $(RERUN_ARGS)

	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Head_Detector_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Interactive_Neuroscope.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/LLaMA.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/No_Position_Experiment.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Othello_GPT.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Qwen.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Santa_Coder.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Stable_Lm.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/SVD_Interpreter_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Tracr_to_Transformer_Lens_Demo.ipynb $(RERUN_ARGS)

	# Contains failing cells

	# Causes CI to hang
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Activation_Patching_in_TL_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Attribution_Patching_Demo.ipynb $(RERUN_ARGS)
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Grokking_Demo.ipynb $(RERUN_ARGS)

test-pr:
	$(MAKE) unit-test
	$(MAKE) docstring-test
	$(MAKE) acceptance-test
	$(MAKE) integration-test

test:
	$(MAKE) unit-test
	$(MAKE) integration-test
	$(MAKE) acceptance-test
	$(MAKE) benchmark-test
	$(MAKE) docstring-test
	$(MAKE) notebook-test

docs-hot-reload:
	$(RUN) docs-hot-reload

build-docs:
	$(RUN) build-docs


# script to set the version in pyproject.toml
define PY_VERSION_SET
import os, re, pathlib, sys
ver = os.environ.get("VERSION")
if not ver:
    sys.exit("VERSION env-var is missing. usage: make version-set VERSION=1.2.3")
path = pathlib.Path("pyproject.toml")
text = path.read_text()
pattern = re.compile(r'^(\s*version\s*=\s*")([^"]*)(")', flags=re.M)
updated = pattern.sub(lambda m: f'{m.group(1)}{ver}{m.group(3)}', text, count=1)
path.write_text(updated)
print(f"Set version to {ver} in {path}")
endef
export PY_VERSION_SET


# Usage: make version-set VERSION=1.2.3
.PHONY: version-set
version-set:
	@python -c "$$PY_VERSION_SET"


# Usage: make version-reset
.PHONY: version-reset
version-reset:
	$(MAKE) version-set VERSION=0.0.0

