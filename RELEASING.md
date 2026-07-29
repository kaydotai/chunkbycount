# Releasing

Releases are built by GitHub Actions and published to PyPI through Trusted
Publishing. Do not store a long-lived PyPI token in the repository.

## Access

- Configure PyPI to trust `.github/workflows/publish.yml` in the
  `kaydotai/chunkbycount` repository with the `pypi` environment.
- Require a trusted reviewer for deployments to the GitHub `pypi` environment.
- Keep at least one recovery method for every PyPI project owner.

## Prepare

1. Choose the release version according to Semantic Versioning.
2. Replace `Unreleased` in `CHANGELOG.md` with the version and release date.
3. Set the same version in `pyproject.toml`.
4. Review the complete diff from the previous release tag.
5. Confirm the GitHub `main` branch is clean, current, and passing CI.

The Git tag must be the project version prefixed with `v`, for example
`v0.1.0`.

## Verify

From a clean checkout:

```bash
poetry sync --with dev
poetry check
poetry run ruff check .
poetry run pytest -q
poetry build
pipx run twine check dist/*
```

Install the wheel into a fresh virtual environment and run a no-network import
smoke test:

```bash
python3 -m venv /tmp/chunkbycount-release-check
/tmp/chunkbycount-release-check/bin/pip install dist/*.whl
/tmp/chunkbycount-release-check/bin/python -c \
  "from chunkbycount import StructuredListExtractor, chunk_by_count"
```

The optional live smoke test requires a user-provided model key. Use only
synthetic, non-sensitive input and record the provider and model identifier.

## Publish

1. Commit and push the release metadata.
2. Wait for every required `main` check to pass.
3. Create and push the signed release tag.
4. Publish the corresponding GitHub release.
5. Review and approve the `pypi` environment deployment. The workflow checks
   that the Git tag matches `project.version`, rebuilds the distributions, and
   publishes them through Trusted Publishing.
6. Verify the PyPI metadata, provenance, wheel contents, README rendering, and
   installation in a clean environment.

Do not reuse artifacts rebuilt after approval; publish the files that were
reviewed and tested.
