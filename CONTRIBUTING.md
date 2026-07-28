# Contributing

Thanks for contributing to `chunkbycount`.

## Development Setup

1. Install Python 3.11 or newer.
2. Install dependencies:

```bash
poetry install --with dev
```

3. Run checks:

```bash
poetry run ruff check .
poetry run pytest -q
```

## Pull Request Guidelines

- Keep PRs scoped and explain motivation in the description.
- Include tests for behavior changes.
- Update documentation when changing public APIs or configuration.
- Do not commit secrets, API keys, or local `.env` files.
- Prefer reproducible scripts over notebook-only workflows.

## Live Model Tests

- Use synthetic, non-sensitive inputs.
- Document the provider and model in the PR without committing credentials or
  generated results.

## Commit Style

- Use clear, imperative commit messages.
- Mention the affected area first (for example: `chunking: prefer record boundaries`).

## Questions

Open an issue for discussion before large refactors or interface changes.
