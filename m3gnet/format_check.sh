#!/bin/bash

# Format the entire directory, ignoring specific rules
autopep8 --in-place --recursive --max-line-length 120 --ignore E201,E221,E225,E226,E241,W605 .

# Check code style
flake8 . --max-line-length 120 --ignore E201,E221,E225,E226,E241,W605

echo "Code formatting and checking completed!"
