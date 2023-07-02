# Makefile

# Set the Python interpreter (change as needed)
PYTHON = python

# Default target
default: install

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Clean and remove installed dependencies
clean-all:
	$(PYTHON) -m pip uninstall -y -r requirements.txt

.PHONY: default install clean-all
