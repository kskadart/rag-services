.PHONY: help lint format type-check test build run stop logs clean

help:
	@echo "Available commands:"
	@echo "  make lint        - Run ruff linter"
	@echo "  make format      - Run black formatter"
	@echo "  make type-check  - Run mypy type checker"
	@echo "  make check       - Run all checks (lint, format, type-check)"
	@echo "  make test        - Run tests"
	@echo "  make build       - Build Docker image"
	@echo "  make run         - Run service locally"
	@echo "  make stop        - Stop service"
	@echo "  make logs        - Show service logs"
	@echo "  make clean       - Remove Docker artifacts"

lint:
	uv run ruff check src tests

format:
	uv run black src tests

format-check:
	uv run black --check src tests

type-check:
	uv run mypy src

check: lint format-check type-check

test:
	uv run pytest tests -v

build:
	docker build -f docker/Dockerfile.cpu -t embedding-service:cpu .

run:
	docker compose -f ../../compose.cpu.yml up -d embedding-service

stop:
	docker compose -f ../../compose.cpu.yml down

logs:
	docker compose -f ../../compose.cpu.yml logs -f embedding-service

clean:
	docker compose -f ../../compose.cpu.yml down -v --rmi local
	docker image prune -f

