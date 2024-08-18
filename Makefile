# Variables
PYTHON ?= python3

# Targets

.PHONY: run
run: 
	@pixi run $(PYTHON) src/main.py
	
.PHONY: clean
clean:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + -print
	@find . -type f -name "*.pyc" -exec rm -f {} + -print
	@find . -type f -name "*.pyo" -exec rm -f {} + -print
	@find . -type f -name "*~" -exec rm -f {} + -print
	@echo "Clean complete."