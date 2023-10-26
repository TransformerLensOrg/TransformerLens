format:
	poetry run pycln --all . --exclude "__init__.py"
	poetry run isort format .
	poetry run black .

check-format:
	poetry run pycln --check --all . --exclude "__init__.py"
	poetry run isort --check-only .
	poetry run black --check .

unit-test:
	poetry run pytest --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/acceptance

docstring-test:
	poetry run pytest transformer_lens/

notebook-test:
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Main_Demo.ipynb 
	poetry run pytest --nbval-sanitize-with demos/doc_sanitize.cfg demos/Exploratory_Analysis_Demo.ipynb

test:
	make unit-test
	make acceptance-test
	make docstring-test
	make notebook-test

docs-hot-reload:
	poetry run docs-hot-reload

build-docs:
	poetry run build-docs
