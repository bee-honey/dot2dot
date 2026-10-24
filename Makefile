.DEFAULT_GOAL := help
PYTHON ?= .venv/bin/python
PORT ?= 8000

.PHONY: help up down restart status logs
help:
	@echo "make up       Start the local app in the background"
	@echo "make down     Stop the app started by make"
	@echo "make restart  Restart the local app"
	@echo "make status   Show whether the app is running"
	@echo "make logs     Show recent server logs"
	@echo "Optional: make up PORT=8001"

up down restart status logs:
	@$(PYTHON) scripts/dev_server.py $@ --port "$(PORT)"
