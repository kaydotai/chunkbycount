# chunkbycount

[![CI](https://github.com/kaydotai/chunkbycount/actions/workflows/ci.yml/badge.svg)](https://github.com/kaydotai/chunkbycount/actions/workflows/ci.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/kaydotai/chunkbycount/blob/main/LICENSE)

`chunkbycount` extracts long structured lists from documents with LLMs. It
divides source text into small token blocks, estimates how many target items
each block contains, and packs consecutive blocks into extraction windows with
a configurable predicted item limit.

The package is alpha software and has not been published to PyPI yet.

## Why item-count adaptive chunking?

A fixed token window can contain two items or two hundred. Large windows may
fit a model's context but still produce incomplete enumerations; very small
windows increase cost and can split records. `chunkbycount` uses this pipeline:

1. Split the source into small token-bounded blocks, preferring paragraph and
   line boundaries when available.
2. Ask a counting model to estimate target items in each block.
3. Adaptively split any block predicted to exceed the item limit.
4. Greedily pack adjacent blocks under predicted item and source-token limits.
5. Extract each packed window into a Pydantic model.
6. Merge list fields and remove exact duplicates in first-occurrence order.

Aggregation, verification guardrails, and semantic deduplication are available
but opt-in because each adds model calls, cost, and failure modes.

The count is an LLM estimate, not a proof. `K=10` means “predicted to contain at
most ten items”; under-counting can still produce a window with more than ten
real items. The library therefore does not claim guaranteed completeness.

## Evaluation snapshot

We evaluated the planner on
[`driver_schedule_sparse_001`](https://github.com/kaydotai/longlistbench/blob/main/data/transcripts/ocr_gemini/driver_schedule_sparse_001.md),
a public synthetic [LongListBench](https://github.com/kaydotai/longlistbench)
document containing 500 target driver rows across 18 OCR pages. The document
mixes three roster layouts and includes administrative columns that are not
part of the output schema.

| Model | Predicted item limit | Count calls | Extraction calls | Returned rows | Benchmark-exact rows | OCR-faithful rows | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.6-sol` | 5 | 151 | 114 | 500 | 499/500 | 500/500 | 117 s |
| `gemini-3.5-flash` | 10 | 101 | 60 | 500 | 499/500 | 500/500 | 94 s |

Both models copied all 500 rows visible in the released OCR transcript. The
single benchmark mismatch is an OCR/annotation discrepancy: the transcript
shows one date as `07/19/1777`, while the ground truth records `07/19/1977`.
The benchmark-exact column uses LongListBench's canonical ground-truth scorer;
the OCR-faithful column compares against the text actually supplied to the
models.

These are diagnostic single-document results, not a leaderboard submission or
a general completeness guarantee. The GPT limit was tightened from 10 to 5
after the larger setting omitted 14 rows, illustrating the quality/cost
tradeoff this package makes explicit. The Gemini run used native Vertex Express
model calls around the same `chunkbycount` planner; the packaged Gemini helper
currently targets the Gemini Developer API's OpenAI-compatible endpoint.

### Inherited-context limitation

Adaptive splitting does not automatically repeat page or section headers in
every child chunk. If a record inherits a required field from a header that is
absent from its extraction window, a prompt cannot recover that missing
evidence reliably. We observed this on LongListBench's 998-row
`ifta_return_schedule_002` case. For such documents, first create
self-contained textual components, propagate the required header context, or
use a larger component boundary.

## Installation

To install from source before the first PyPI release:

```bash
git clone https://github.com/kaydotai/chunkbycount.git
cd chunkbycount
poetry install --with dev
```

After the first PyPI release:

```bash
pip install chunkbycount
```

Semantic deduplication has heavier numerical dependencies and is installed
separately:

```bash
pip install "chunkbycount[semantic-dedup]"
```

Python 3.11 or newer is supported.

## Quick start

Set `OPENAI_API_KEY` and choose model names available to your account.

```python
import os

from pydantic import BaseModel, Field

from chunkbycount import StructuredListExtractor, configure_from_environment


configure_from_environment()  # Loads .env when present.
extraction_model = os.environ["EXTRACTION_MODEL"]
counting_model = os.getenv("COUNTING_MODEL", extraction_model)


class LineItem(BaseModel):
    item_id: str
    description: str | None = None
    amount: float | None = None


class LineItems(BaseModel):
    items: list[LineItem] = Field(default_factory=list)


def build_prompt(text: str) -> str:
    return (
        "Extract every line item. Preserve identifiers exactly and do not "
        f"invent values.\n\n## Source\n{text}"
    )


extractor = StructuredListExtractor(
    model_class=LineItems,
    extraction_model_name=extraction_model,
    counting_model_name=counting_model,
    extraction_system_prompt="Return all requested items as structured data.",
    extraction_prompt_builder=build_prompt,
    max_count_per_chunk=10,
    token_per_chunk=250,
    max_tokens_per_window=8_000,
)

result = extractor.extract("...document text...")
print(result.model_dump())
```

Use `await extractor.aextract(...)` from asynchronous applications. Input can
also be UTF-8 bytes or a list of dictionaries containing `document_text`.
The output wrapper should normally have one list field. If it has more than
one, set `list_field_name`; list fields are concatenated while non-list fields
are retained from the first window.

### Gemini Developer API through the OpenAI-compatible endpoint

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
EXTRACTION_MODEL=...
```

Call `configure_from_environment()` before constructing the extractor. The
helper uses public Agents SDK configuration and does not copy the Gemini key
into `OPENAI_API_KEY`.

This helper expects a Gemini Developer API key accepted by the public
Generative Language endpoint. A Vertex AI Express API key
(`VERTEX_AI_API_KEY`) is a different credential and is not accepted there.
Vertex's OpenAI-compatible endpoint uses Google Cloud OAuth credentials; the
library does not manage that token refresh. Use a supported OpenAI-compatible
credential/endpoint pair or provide a separate integration for Vertex.

### Choosing models

`chunkbycount` does not keep a model allowlist. Supply provider model IDs that
are available to your credentials through the configured endpoint and support
text input and structured output. If `counting_model_name` is omitted, the
extraction model is used for counting too.

Current practical choices are:

| Provider | Extraction model | Counting model |
|---|---|---|
| OpenAI | [`gpt-5.6-sol`](https://developers.openai.com/api/docs/models/gpt-5.6-sol) for maximum quality, [`gpt-5.6-terra`](https://developers.openai.com/api/docs/models/gpt-5.6-terra) for balance, or [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna) for high-volume workloads | `gpt-5.6-luna` is the usual lower-cost choice; use Terra or Sol when counting accuracy matters more than cost |
| Gemini Developer API | [`gemini-3.6-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash) or the benchmarked [`gemini-3.5-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash) | [`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) for lower-cost counting, or `gemini-3.5-flash` when accuracy matters more |

The benchmark above used the same model for counting and extraction:
`gpt-5.6-sol` or `gemini-3.5-flash`. The other combinations are
provider-documented choices, not benchmark results from this repository.
Availability and model IDs change, so consult the live
[OpenAI](https://developers.openai.com/api/docs/models) or
[Gemini](https://ai.google.dev/gemini-api/docs/models) catalog when configuring
a deployment.

## Important options

| Option | Default | Effect |
|---|---:|---|
| `extraction_model_name` | required | Provider model ID used for structured extraction |
| `counting_model_name` | extraction model | Provider model ID used for count probes; see the choices above |
| `max_count_per_chunk` | `10` | Predicted item limit per extraction window |
| `token_per_chunk` | `250` | Source-token size of initial counting blocks |
| `max_tokens_per_window` | `8000` | Source-token limit for a packed window |
| `batch_size` | `20` | Maximum concurrent calls within each stage |
| `use_aggregation` | `False` | Verify/enrich outputs with adjacent context |
| `use_guardrails` | `False` | Run an additional verification pass |
| `semantic_deduplication` | `False` | Use embeddings and an LLM on similar items |
| `deduplicate_items` | `True` | Remove exact duplicate output objects |

If identical records are legitimate occurrences in your domain, set
`deduplicate_items=False`. A task-specific `deduplication_key` can be supplied
when identity is well-defined.

Supplying aggregation or guardrail configuration enables that stage when its
`use_...` flag is omitted. An explicit `False` always disables the stage.

`max_tokens_per_window` counts source text only. It does not include the system
prompt, schema, or response budget. Unknown model names use `cl100k_base` as an
approximate tokenizer, so non-OpenAI providers should leave a safety margin.

## Failure behavior

The library retries transient provider failures a small, bounded number of
times. If any count, extraction, aggregation, or guardrail call still fails,
the operation raises `BatchExecutionError` with the failed task indexes. It
does not silently return a partial extraction. Calls use a sliding concurrency
limit and stop scheduling new work as soon as a task fails.

An implausibly high count estimate is isolated as its own token-bounded window
and emits `CountEstimateWarning`; it does not abort the document. If every
initial block is estimated at zero, the planner warns and extracts the original
small blocks rather than silently combining them into one large window. This
fallback can require one extraction call per initial block and substantially
increase cost, so treat `CountEstimateWarning` as an operational signal.
`ChunkPlanningError` is reserved for tokenizer round-trip or safe-boundary
failures that prevent lossless planning.

## Environment configuration

The library recognizes these optional settings:

| Variable | Default | Purpose |
|---|---:|---|
| `CHUNKBYCOUNT_RUNNER_TIMEOUT_S` | `300` | Timeout for count, extraction, aggregation, and guardrail calls |
| `CHUNKBYCOUNT_DEDUP_TIMEOUT_S` | `120` | Timeout for an LLM semantic-deduplication pass |
| `CHUNKBYCOUNT_DEDUP_SIMILARITY_THRESHOLD` | `0.95` | Cosine threshold in the inclusive range `0`–`1` |
| `CHUNKBYCOUNT_EMBED_TIMEOUT_S` | `120` | Embedding-client timeout |
| `CHUNKBYCOUNT_EMBED_MAX_RETRIES` | `2` | Embedding-client retry count; `0` is valid |
| `CHUNKBYCOUNT_ENABLE_TRACING` | `false` | Enable Agents SDK tracing |

Invalid configured values raise `ValueError` instead of silently falling back
to defaults. `EXTRACTION_MODEL` and `COUNTING_MODEL` are used only by the
examples; they are not read by the library.

## Privacy and tracing

Document fragments and schemas are sent to the configured model provider.
Optional semantic deduplication also sends serialized extracted items to an
embedding endpoint and a model.

Document text is untrusted model input. It can contain prompt-injection text
that influences both the count probe and extraction calls. The anomalous-count
fallbacks above catch only obvious extremes, not plausible under-counts.
Applications processing adversarial documents should add independent coverage
checks and avoid treating model output as trusted data.

Agents SDK tracing is disabled by default. Set
`CHUNKBYCOUNT_ENABLE_TRACING=true` only if sending trace data is appropriate
for the documents being processed. Full-context aggregation sends the entire
document once per extraction window and should be enabled deliberately.

## Development

```bash
poetry install --with dev
poetry run ruff check .
poetry run pytest -q
poetry build
```

See
[CONTRIBUTING.md](https://github.com/kaydotai/chunkbycount/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/kaydotai/chunkbycount/blob/main/SECURITY.md),
and
[RELEASING.md](https://github.com/kaydotai/chunkbycount/blob/main/RELEASING.md)
before contributing or releasing.

## License

The code is available under the [MIT License](https://github.com/kaydotai/chunkbycount/blob/main/LICENSE).
