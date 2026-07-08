# AI Usage

## AI Tools Used
- GitHub Copilot Chat (GPT-5.3-Codex) in VS Code.

## What I Asked AI To Do
- Review current retrieval/chunking quality.
- Propose and implement semantic chunking improvements.
- Preserve page source attribution when chunks span multiple pages.
- Update retrieval/source formatting and UI compatibility.
- Validate syntax and run local smoke tests.

## What I Accepted
- Moving from page-local chunking to semantic block packing.
- Adding page-range metadata (`page_start`, `page_end`) and `source_spans`.
- Updating source rendering to support page ranges.
- Lowering chunk target/overlap defaults for better retrieval precision.

## What I Rejected or Adjusted
- Initial heading heuristics that produced weak section titles (for example, generic labels like "Equation" or table-of-contents entries).
- A source schema variant where `page` could be `null`; this was adjusted to keep backward compatibility for clients expecting an integer page.

## Example of Wrong/Incomplete AI Output
- Early AI-generated heading inference overfit textbook noise and produced unhelpful section titles.
- Early AI change set `AskSource.page` to nullable for range citations, which risked breaking clients that expected `page` to always exist.

How it was corrected:
- Tightened heading detection rules and excluded generic/toC-like headings.
- Restored `page` as a required integer while adding `page_start`/`page_end` for richer attribution.

## How Final Implementation Was Verified
- Compiled affected modules with `python -m py_compile`.
- Ran chunking smoke test on a real extracted JSON artifact.
- Checked generated chunk metadata for:
  - valid page ranges,
  - non-empty text,
  - source spans,
  - improved chunk count versus page-level baseline.

## What I Would Improve With More Time
- Add deterministic mock mode for full `/ask` E2E tests without external LLM keys.
- Add hybrid retrieval and optional reranking.
- Improve heading detection with document-structure signals beyond simple heuristics.
- Add regression tests for source attribution and citation formatting.
- Add a small benchmark suite for retrieval quality across representative textbook queries.
