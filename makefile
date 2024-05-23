format:
	poetry run pycln --all . --exclude "__init__.py"
	poetry run isort format .
	poetry run black .

check-format:
	poetry run pycln --check --all . --exclude "__init__.py"
	poetry run isort --check-only .
	poetry run black --check .

unit-test:
	poetry run pytest tests/unit

integration-test:
	poetry run pytest tests/integration

acceptance-test:
	poetry run pytest tests/acceptance

coverage-report-test:
	poetry run pytest --cov=transformer_lens/ --cov-report=html --cov-branch tests/unit tests/integration tests/acceptance

docstring-test:
	poetry run pytest transformer_lens/

notebook-test:
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/BERT.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Exploratory_Analysis_Demo.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Main_Demo.ipynb

	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Head_Detector_Demo.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Interactive_Neuroscope.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/LLaMA.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/No_Position_Experiment.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Othello_GPT.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Qwen.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Santa_Coder.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Stable_Lm.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/SVD_Interpreter_Demo.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Tracr_to_Transformer_Lens_Demo.ipynb

	# Contains failing cells

	# Causes CI to hang
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Activation_Patching_in_TL_Demo.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Attribution_Patching_Demo.ipynb
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Grokking_Demo.ipynb

test:
	make unit-test
	make acceptance-test
	make docstring-test
	make notebook-test

docs-hot-reload:
	poetry run docs-hot-reload

build-docs:
	poetry run build-docs
