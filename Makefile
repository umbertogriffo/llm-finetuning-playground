.PHONY: check install setup tidy check-formatting test clean

check:
	which pip3
	which python3
	which uv

install:
	uv sync

install_pre_commit:
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-commit

setup: install install_pre_commit

tidy:
	uv run ruff format --exclude=.venv .
	uv run ruff check --exclude=.venv . --fix

check-formatting:
	uv run ruff format . --check

clean:
	echo "Cleaning Poetry environment..."
	rm -rf .venv
	echo "Cleaning all compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	echo "Cleaning the cache..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache

finetune:
	echo "Finetuning LLM"
	python src/finetune.py
