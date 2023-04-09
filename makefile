test:
	make unit-test
	make acceptance-test
	make documentation-test

unit-test:
	poetry run pytest -v --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/acceptance

documentation-test:
	poetry run pytest --nbmake demos/*ipynb