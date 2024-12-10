# Phony targets for commands that don't create files
.PHONY: all clean data eval help

# Default target
all: data eval

# Help command to show available targets
help:
	@echo "Available targets:"
	@echo "  make data  - Run get_datasets.py to fetch datasets"
	@echo "  make eval  - Run eval.py for evaluation"
	@echo "  make all      - Run both datasets and evaluate targets"
	@echo "  make clean    - Remove generated files and cached data"

# Target to get datasets
data:
	@echo "Fetching datasets..."
	python get_datasets.py

# Target to run evaluation
eval:
	@echo "Running evaluation..."
	python eval.py

# Clean generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache
	# Add any other generated files or directories to clean