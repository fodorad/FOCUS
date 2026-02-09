.PHONY: install dev test lint format clean ci

install:
	pip install "focus[dev]"

dev:
	pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .

test:
	coverage run -m unittest discover -s tests -v
	coverage report --fail-under=85 --show-missing
	coverage html

ci: format lint test

clean:
	rm -rf .venv coverage_html dist/ .pytest_cache/ tmp/
