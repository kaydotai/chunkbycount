# Releasing

No package or repository visibility change should be made until the owner
explicitly approves the public release.

## First-release prerequisites

- Confirm that Kay.ai owns or has permission to publish every tracked file.
- Remove production-only prompts, customer data, credentials, internal URLs,
  and proprietary fixtures.
- Confirm the public repository has no archived paper history, obsolete
  branches, tags, pull requests, releases, or workflow logs that should remain
  private.
- Verify the GitHub description, topics, README rendering, and detected MIT
  license before changing repository visibility.
- Confirm the `chunkbycount` name on PyPI and TestPyPI immediately before use.
- Configure a PyPI Trusted Publisher for `.github/workflows/publish.yml` and
  add required reviewers to the GitHub `pypi` environment. Private-repository
  plans may expose those controls only after the visibility change. Do not
  store a long-lived PyPI token in the repository.
- Before making the GitHub repository public, enable branch protection and
  Dependabot alerts. Immediately after the visibility change, verify secret
  scanning and push protection, then enable private vulnerability reporting.

## Build and verify

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

## Version and metadata

For the first public release, replace `Unreleased` in `CHANGELOG.md` with the
release version and date. Keep these values aligned:

- `project.version` in `pyproject.toml`
- Git tag (for example, `v0.1.0`)

## Publication sequence

1. Upload the verified distribution to TestPyPI and install it in a fresh
   environment.
2. Obtain explicit approval for repository visibility and PyPI publication.
3. Publish the GitHub repository and create the signed release tag.
4. Publish the GitHub release. The `Publish to PyPI` workflow verifies that
   its tag matches `project.version`, rebuilds and checks the distribution,
   then publishes through Trusted Publishing after environment approval.
5. Verify the PyPI metadata, wheel contents, README rendering, and clean
   installation.

Do not reuse artifacts rebuilt after approval; publish the files that were
reviewed and tested.
