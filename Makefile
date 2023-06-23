.PHONY: all local_setup format lint tests integration_tests help

all: help

local_setup:
	./scripts/local_setup.sh

format:
	pdm run ruff --select I --fix .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$')

lint lint_diff:
	pdm run mypy $(PYTHON_FILES)
	pdm run ruff .

TEST_FILE ?= tests/unit_tests/

tests:
	pdm run pytest $(TEST_FILE) -s

integration_tests:
	pdm run pytest tests/integration_tests -s

help:
	@echo '----'
	@echo 'local_setup					- setup local dev environment'
	@echo 'format						- run code formatters'
	@echo 'lint							- run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_tests            - run integration tests'
