RUN := uv run

# Rerun args for flaky tests (httpx timeouts during HF Hub downloads)
# Remove this line when no longer needed
RERUN_ARGS := --reruns 2 --reruns-delay 5

dep:
	uv sync

format:
	$(RUN) pycln --all . --exclude "__init__.py"
	$(RUN) isort format .
	$(RUN) black .

check-format:
	$(RUN) pycln --check --all . --exclude "__init__.py"
	$(RUN) isort --check-only .
	$(RUN) black --check .

unit-test:
	$(RUN) pytest tests/unit $(RERUN_ARGS)

integration-test:
	$(RUN) pytest tests/integration $(RERUN_ARGS)

acceptance-test:
	$(RUN) pytest tests/acceptance $(RERUN_ARGS)

benchmark-test:
	$(RUN) pytest tests/benchmarks $(RERUN_ARGS)

coverage-report-test:
	$(RUN) pytest --cov=transformer_lens/ --cov-report=html --cov-branch tests/integration tests/benchmarks tests/unit tests/acceptance $(RERUN_ARGS)

docstring-test:
	$(RUN) pytest transformer_lens/ $(RERUN_ARGS)

notebook-test:
	$(RUN) pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/BERT.ipynb $(RERUN_ARGS)
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

test:
	make unit-test
	make integration-test
	make acceptance-test
	make benchmark-test
	make docstring-test
	make notebook-test

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

