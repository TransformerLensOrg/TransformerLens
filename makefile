format:
	poetry run python -m pycln --all . --exclude "__init__.py"
	poetry run python -m isort format .
	poetry run python -m black .

check-format:
	poetry run python -m pycln --check --all . --exclude "__init__.py"
	poetry run python -m isort --check-only .
	poetry run python -m black --check .

test:
	make unit-test
	make acceptance-test
	make documentation-test

unit-test:
	poetry run pytest -v --typeguard-packages=transformer_lens --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --typeguard-packages=transformer_lens --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/acceptance

documentation-test:
	poetry run pytest --nbmake demos/*ipynb
