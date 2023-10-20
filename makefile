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
	poetry run pytest transformer_lens/ --doctest-modules --doctest-plus

notebook-test:
	poetry run pytest demos/Exploratory_Analysis_Demo.ipynb --nbval

test:
	make unit-test
	make acceptance-test
	make docstring-test
	make notebook-test
