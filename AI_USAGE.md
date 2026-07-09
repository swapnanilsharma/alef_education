# AI Usage Summary

## Overview
I used AI coding assistance to accelerate implementation while keeping engineering decisions and final quality control manual. The workflow was consistent throughout the project:

1. inspect the existing code and requirements,
2. ask AI to propose implementation options,
3. review each proposal critically,
4. accept only changes that matched functional and compatibility requirements,
5. verify behavior with local checks and sample runs.

AI was most helpful for iteration speed, especially when refining chunking and source attribution.

## AI Tools Used
- GitHub Copilot Chat in VS Code.

## What I Asked AI To Do
- review current chunking and retrieval behavior,
- propose and implement semantic chunking improvements,
- preserve page-source attribution when chunks span pages,
- update retrieval, response formatting, and UI compatibility,
- draft architecture and usage documentation,
- generate a local chunking evaluation script.

## What I Accepted
- moving from page-local chunking to semantic block packing,
- adding chunk provenance fields (`page_start`, `page_end`, `source_spans`),
- keeping `page` in API responses for backward compatibility,
- reducing chunk size/overlap defaults for better retrieval precision,
- adding a deterministic local evaluation script for chunk metadata checks.

## What I Rejected or Adjusted
- early heading heuristics that produced weak titles (for example generic labels and table-of-contents artifacts),
- an early response shape where `page` could be null,
- wording in draft documentation that was too implementation-heavy for an interview submission.

## Example of Incorrect or Incomplete AI Output
One meaningful correction was in source response compatibility.

AI initially suggested making `page` nullable and using only `page_start`/`page_end` for range citations. That was structurally valid but risky for clients expecting `page` to always be present.

I corrected this by:
- keeping `page` required,
- setting `page = page_start`,
- adding `page_start` and `page_end` as richer metadata for range-aware clients.

Another correction was heading inference. Early heuristic output overfit textbook noise and produced low-value section titles. I tightened heading rules and excluded generic or table-of-contents style headings.

## How I Verified the Final Implementation
- compiled affected modules with `python -m py_compile`,
- executed the chunking evaluation script on a real extracted JSON artifact,
- reviewed sample chunk outputs for:
  - valid page ranges,
  - non-empty chunk text,
  - populated source spans,
  - expected cross-page chunk behavior,
  - improved chunk granularity versus page-level baseline,
- manually inspected the ask response path to confirm backward-compatible source fields.

## What I Would Improve With More Time
- add deterministic mock mode for full `/ask` end-to-end tests without external LLM dependencies,
- add authentication and authorization (for example JWT-based API auth with role-based access),
- add keyword-search retrieval (for example BM25) and combine it with vector search as a hybrid approach,
- add hybrid retrieval and optional reranking,
- add regression tests specifically for citation formatting and source attribution,
- add a retrieval benchmark set with representative textbook queries,
- improve section detection with stronger document-structure signals,
- capture clearer before/after retrieval quality metrics.

## Final Assessment
AI improved development velocity and idea exploration, but correctness depended on manual review, targeted corrections, and validation. The best value came from using AI as an accelerator, not as an autonomous decision-maker.
