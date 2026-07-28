# Changelog

All notable changes to this project are documented here. The project follows
Semantic Versioning.

## [Unreleased]

### Added

- Public `StructuredListExtractor`, chunk-planning, exact-deduplication, and
  provider-configuration APIs.
- Explicit count and source-token budgets with adaptive subdivision of blocks
  predicted to exceed the item limit.
- Paragraph- and line-boundary preference within the hard token limit.
- Python 3.11–3.14 support and PyPI-ready project metadata.
- PEP 561 typing marker for downstream type checkers.
- A public LongListBench evaluation snapshot with call counts, timing,
  source-faithful results, and documented limitations.
- Unit coverage for Unicode boundaries, adaptive planning, failure
  propagation, configuration validation, deduplication, and tracing defaults.

### Changed

- Made aggregation, guardrails, and semantic deduplication opt-in.
- Removed the legacy duplicate-enrichment module.
- Replaced punctuation-insensitive deduplication with deterministic exact
  matching.
- Reduced the runtime dependency set; numerical dependencies now live in the
  `semantic-dedup` extra.
- Switched Gemini configuration from private SDK internals and environment
  mutation to public Agents SDK APIs.
- Disabled Agents SDK tracing by default.
- Replaced barrier batches with fail-fast sliding concurrency.

### Fixed

- Enforced the predicted item cap when an initial token block is over budget.
- Raised stage-specific errors instead of silently dropping failed model calls.
- Removed overlap that could cause an extraction window to violate its
  predicted item limit.
- Preserved Unicode text across token boundaries.
- Prevented aggregation retries from mutating message history.
- Made explicit optional-stage opt-outs override supplied stage configuration.
- Added conservative fallbacks and warnings for implausible high and all-zero
  count estimates.
- Isolated blocks that cannot meet a count-derived token target without
  corrupting Unicode instead of aborting the document.
- Made invalid timeout, retry, threshold, and tracing environment values fail
  loudly.
- Closed unstarted model coroutines when timeout configuration is invalid and
  preserved task cancellation through the batch runner.
