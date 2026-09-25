# Source selection

Prepared from selected application source, with fresh public Git history.
The public snapshot was reviewed and hardened on September 24, 2026.

Included: document parsing, event reconciliation, canonical dataset construction,
the main and first-round models, player matching, weather adjustments,
recommendation/price snapshot utilities, and the offline model example.

Excluded: provider responses, source PDFs, personal betting records, account
screenshots, secrets, live configuration, internal session notes, deployment
jobs, model outputs, and prior Git history. Full application interfaces and
data acquisition orchestration are outside this source selection.

Publication adaptations:

- Default budgets and all demonstration amounts are fictional.
- Removed promotional performance comments and a fixed historical ROI banner.
- Corrected the softmax docstring to describe the implementation; numerical
  stability does not establish calibration.
- Document extraction and event reconciliation read `OPENAI_API_KEY` from the
  environment instead of a local credential file. The old `--keys` option is
  intentionally absent.
- Added synthetic contract tests, public documentation, and Python 3.10/3.12 CI.

Scoring formulas are otherwise retained. The full model commands expect local
data and may write generated files. The documented offline entry point does not
invoke acquisition, train a model, write records, or submit transactions.

## Publication hardening

All financial fixtures are invented. Source comments retain technical rationale
without operational incident narratives. Public Git authorship uses the account
handle and a noreply address. The publication file allowlist, pinned CI actions,
and full-history secret scan enforce the boundary described in
[the publication policy](PUBLICATION_POLICY.md). Public CLAUDE.md and AGENTS.md
contain only general maintenance rules, not private session history.

API errors omit response details, document downloads validate the provider HTTPS
domain including redirects, and the non-security use of the identifier hash is
explicit. Hard-coded historical return metadata is omitted; performance metrics
must be computed from user-supplied evaluation data.
