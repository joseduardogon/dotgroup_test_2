.DEFAULT_GOAL := help
.PHONY: help install check-config chat ask demo test test-live lint format typecheck check docker-build docker-chat

help: ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	poetry install
	poetry run pre-commit install

check-config: ## Validate .env / environment without calling any API
	poetry run pychat check

chat: ## Start the interactive assistant
	poetry run pychat chat

ask: ## Ask one question: make ask Q="Como criar uma lista em Python?"
	poetry run pychat ask "$(Q)"

demo: ## Run every question in examples/questions.txt (needs OPENAI_API_KEY)
	@while IFS= read -r q; do printf "\n### %s\n" "$$q"; poetry run pychat ask --raw "$$q"; done < examples/questions.txt

test: ## Offline test suite with coverage (no API key needed)
	poetry run pytest

test-live: ## One real, paid request to OpenAI (needs OPENAI_API_KEY)
	poetry run pytest -m live --no-cov

lint: ## Static analysis (ruff)
	poetry run ruff check .
	poetry run ruff format --check .

format: ## Auto-format and apply safe fixes
	poetry run ruff check --fix .
	poetry run ruff format .

typecheck: ## Strict type checking (mypy)
	poetry run mypy

check: lint typecheck test ## Everything CI runs

docker-build: ## Build the container image
	docker build -t dotgroup-test-2:latest .

docker-chat: ## Interactive chat in a container (reads .env)
	docker compose run --rm chat
