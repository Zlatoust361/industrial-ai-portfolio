# rag/ — week 19: RAG over factory regulations

Status: **first implementation pass**, part of
[`course/ROADMAP-WEEKS-13-24.md`](../course/ROADMAP-WEEKS-13-24.md)'s
volume 2 (not wired into `course.yaml`/`validate_course.py` — see that
roadmap's open question #1). Lives at the repo root, next to
`tracking/`/`depth_fusion/`/`peft/`/`fleet/`/`captioning/`/
`active_learning/`, not under `course/weeks/`.

## Idea

Week 11's `assistant/` executes operator COMMANDS — it has no way to
answer "WHY" with a citation to a real document. On a real production
line the operator/technician has a whole shelf of regulations/runbooks,
and the question is almost always "what does the instruction say" — this
week adds that: ingest real Markdown documents (`docs/runbook.md`/
`docs/architecture.md`, already in this repository), retrieve the
relevant section by embedding similarity (reusing CLIP's text branch,
not a new model), and answer with the regulation's own text PLUS which
document and section it came from — with an honest, explicit "not found"
when nothing clears the relevance bar, never a confident-sounding guess.

## What's new here vs. reused from other weeks

| Piece | Status |
|---|---|
| `ingestion.py` | **New** — splits a real Markdown document into chunks by header structure (`##`/`###`), not a fixed character window, so a citation actually names a real section. |
| `index.py` | **New** — `RagIndex`, a plain in-memory array of chunks + embeddings (no vector-DB dependency at this scale). |
| `retriever.py` | **New** — cosine-similarity top-k with an explicit relevance threshold — `retrieve()` can and does return nothing. |
| `answer.py` | **New** — extractive (quotes the regulation verbatim + cites it), not generative — see below for why. |
| CLIP text embeddings | Reused by import: `src.models.clip.CLIPLoader`/`CLIPInference` (weeks 01/02's text branch), only in `rag.integration`'s default path — not a new embedding model. |
| `assistant/` intent wiring | **New integration**, small and additive: `assistant/schemas.py` gains `AskRunbookParams`/`"ask_runbook"`, `assistant/llm_parser.py` gains one local matcher, `assistant/action_executor.py` gains one handler + one optional constructor parameter. No existing intent's behavior changed. |

## Pipeline

```
rag/scripts/run_rag_demo.py                    -- ingests the REAL
        │  docs/runbook.md + docs/architecture.md (not synthetic text)
        ▼
ingestion.ingest_markdown_documents()            -- split by ##/### headers
        ▼
index.RagIndex.build()                              -- embed every chunk
        │  (real CLIP text encoder -- the one real network call this week makes)
        ▼
retriever.retrieve()                                  -- cosine similarity,
        │  top_k, min_similarity -- empty if nothing clears the bar
        ▼
answer.build_answer()                                   -- quote the top
                                                            chunk(s) + cite them,
                                                            or an honest "not found"
        │
        ▼
    assistant/action_executor.py::_handle_ask_runbook  -- the SAME
    integration.answer_question_with_default_index(), wired into a new
    "ask_runbook" intent operators can actually type in the chat UI
    (see "Integration into assistant/" below).
```

## Why extractive, not generative

`rag.answer.build_answer` quotes the retrieved chunk's text VERBATIM and
cites it, instead of asking an LLM to paraphrase or compose across
chunks. For "what does the instruction say" — this week's actual framing
— a literal quote is MORE trustworthy than a generative rewrite, not
less: a paraphrase can subtly misstate a safety-relevant step. It also
keeps this week's new retrieval/grounding logic fully deterministic and
testable without a second generative network call (week 15 already adds
one, for captioning). The documented next step (see Honesty note) is
routing a multi-chunk answer through `assistant/llm_parser.py`'s existing
local/cloud split to COMPOSE a single coherent answer from more than one
source — left undone here, not because it's hard, but because the
extractive version already fully demonstrates this week's actual
skill (retrieval + honest grounding), and adding a second generative
call would mix two concerns in one first pass.

## Integration into `assistant/` — small, additive, nothing else changed

- `assistant/schemas.py`: new `AskRunbookParams(question: str)`, registered
  as `"ask_runbook"` in `INTENT_PARAM_SCHEMAS` — `KNOWN_INTENTS` picks it
  up automatically (it's derived from that registry's keys).
- `assistant/llm_parser.py`: new `_match_ask_runbook` + `_ASK_RUNBOOK_TRIGGER`
  — narrow, topic-specific keywords mapped to `docs/runbook.md`'s own 5
  incident sections (camera/quality-gate/disk/TensorRT/thermal), checked
  LAST in `_LOCAL_MATCHERS` so it never shadows a more specific existing
  intent.
- `assistant/action_executor.py`: new `rag_answer_fn` constructor
  parameter (optional, defaults to a LAZY wrapper around
  `rag.integration.answer_question_with_default_index` — constructing an
  `ActionExecutor` never forces a network call, same lazy-load convention
  every `VLMLoader` in this repository already follows) and one new
  `_handle_ask_runbook` method. Every existing constructor call site and
  every existing intent is unchanged — confirmed by running `assistant/`'s
  full pre-existing test suite before and after (156 passed / 1
  pre-existing failure, both times; 168 passed / 1 pre-existing failure
  after adding this week's own new tests on top).

## Honesty note — what's verified for real vs. not yet done

What IS verified for real in this sandbox (22 new `rag/` tests + 12 new
`assistant/` tests, all passing):

- `ingestion.py`: real chunking behavior against the REAL
  `docs/runbook.md` (not a synthetic fixture only) — confirms the real
  document's 5 numbered incident sections are each their own chunk.
- `index.py`/`retriever.py`: exact cosine-similarity arithmetic against a
  fake encoder with known vectors — identical/orthogonal vectors give the
  expected 1.0/0.0 scores, not "roughly plausible"; the empty-result path
  (nothing clears `min_similarity`) is a real, tested outcome, not an
  afterthought.
- `answer.py`: `is_grounded=False` with `NOT_FOUND_TEXT` for an empty
  retrieval; a grounded answer quotes the chunk's text EXACTLY, not a
  paraphrase of it.
- `integration.py`: the injectable path is fully exercised with a fake
  encoder; the default-index cache is confirmed to build the real index
  only ONCE across repeated calls (mocked `RagIndex.build` call count).
- `scripts/run_rag_demo.py`: the real default encoder path is confirmed
  to reach exactly the real `httpx2.ProxyError`/403 network boundary this
  sandbox's other weeks (01/02/09/15/16/17) already document.
- `assistant/`'s new `ask_runbook` intent: `match_locally` recognizes
  real runbook-topic phrasings (camera/quality-gate/disk/TensorRT/thermal,
  in both Russian and English) and correctly does NOT fire for a plain
  `list_defects`-shaped command; `ActionExecutor._handle_ask_runbook`
  with an injected fake `rag_answer_fn` reports both the grounded and
  ungrounded cases as `success=True` (not finding a relevant section is
  a legitimate answer, not a failed command).

**Update (real dev-machine run, reported 2026-10-24)**: `make demo`
reached a real CLIP checkpoint for the first time outside this sandbox
(network reachable there, unlike here) and hit a real crash:
`ValueError: Sequence length must be less than max_position_embeddings
(got sequence length: 1290 and max_position_embeddings: 77)`.
`index.py`'s chunking is deliberately header-based, not token-bounded
(see `ingestion.py`'s own docstring on why), so a long `##` section can
easily exceed CLIP's text encoder's hard 77-token context window — and
`CLIPInference.encode_text` (`src/models/clip.py`) never passed
`truncation=True` to the processor, so instead of silently truncating
it crashed outright. Fixed by adding `truncation=True` to that call
(benefits every week that uses `encode_text`, not just this one, though
weeks 01/02/09/16/17/20's own prompts were always short enough that this
never surfaced there). **Real, now-documented limitation this exposes,
not fully solved by the fix**: a chunk longer than CLIP's 77-token
window is now silently truncated for EMBEDDING purposes — retrieval
only "sees" roughly the first 60-77 tokens of a long section, even
though the full section text is still shown in the citation/answer.
For `docs/runbook.md`/`docs/architecture.md`'s longer sections, content
past the first paragraph is effectively invisible to semantic search.
CLIP's text branch was reused here because it's the same embedding
space every other week already uses (see "which encoder" above), not
because it's well-suited to long-form document retrieval — a real
production RAG system over longer regulations would need either
sub-chunking long sections into token-bounded embedding windows (while
keeping the full section as the citation unit) or a dedicated long-context
text embedding model instead of CLIP's text branch. Neither is done here.

**Still not yet done**: composing a GENERATIVE multi-chunk answer through
`assistant/llm_parser.py`'s existing local/cloud split (see "Why
extractive, not generative" above); PDF/DOCX ingestion (this repository's
real regulations are Markdown — `ingestion.py` only handles that; a real
PDF/DOCX parser is the natural next step once there's a real PDF/DOCX
regulation to parse, not invented ahead of one); the sub-chunking/
long-context fix above; and reindexing when `docs/runbook.md`/
`docs/architecture.md` change on disk (the default index is cached for
the process lifetime, with no invalidation on file change — fine for a
demo, not for a long-running service that might outlive a regulation
update).

## Usage

```bash
cd rag
make setup
make demo                 # ingests docs/runbook.md + docs/architecture.md, asks 3 questions
```

See `rag/Makefile` for each target's exact CLI, and
`course/weeks/week19-rag-factory-docs/MANUAL-RUN.md` for the code-level
walkthrough and known-failure-boundary details.
