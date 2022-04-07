PYTHON_DIRS = tests src
PYTHON_FILES = $(PYTHON_DIRS:=/$(wildcard '*.py')) main.py

all: test lint format

init:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	pylint $(PYTHON_FILES)

format:
	yapf -vv -r -i $(PYTHON_FILES)
