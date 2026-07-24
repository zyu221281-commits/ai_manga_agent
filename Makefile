# 快捷命令

.PHONY: up down logs migrate test lint format

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

migrate:
	python scripts/init_db.py

test:
	python scripts/test_interrupt.py

lint:
	ruff check app

format:
	black app
