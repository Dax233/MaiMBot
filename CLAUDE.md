# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Run Bot**: `python bot.py`
- **Lint**: `ruff check --fix .` or `ruff format .`
- **Run Tests**: `python -m unittest discover -v`
- **Run Single Test**: `python -m unittest src/plugins/message/test.py`

## Code Style
- **Formatting**: Line length 120 chars, use double quotes for strings
- **Imports**: Group standard library, external packages, then internal imports
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use try/except blocks with specific exceptions
- **Types**: Use type hints where possible
- **Docstrings**: Document classes and complex functions
- **Linting**: Follow ruff rules (E, F, B) with ignores E711, E501

When making changes, run `ruff check --fix .` to ensure code follows style guidelines. The codebase uses Ruff for linting and formatting.