#!make

.PHONY: lint
lint:  ## lint python code
	@poetry run isort . --check-only --diff
	@poetry run black . --check --diff
	@poetry run pylint .
	@poetry run bandit -c pyproject.toml -r .
	@poetry run mypy --explicit-package-bases --disallow-untyped-defs .

.PHONY: test
test:  ## test python code
	@poetry run pytest -s --cov --cov-branch .

.PHONY: notebook
notebook:  ## to run jupyter lab in examples
	@poetry run jupyter lab
