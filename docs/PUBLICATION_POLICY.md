# Publication policy

This is a reviewed source portfolio, not an operational backup.

Only files listed in PUBLICATION_FILES.txt may be tracked. Changes to that list
require content review, including comments, docstrings, examples, and fixtures.
Do not copy examples from real transactions: invent the event, participant,
amount, price, and identifiers independently. Public sports reference names are
permitted; personal contact details and account identifiers are not.

Never commit credentials, local environment files, account records, datasets,
model artifacts, screenshots, PDFs, uploads, private instructions, or session
history. Ignore rules are a convenience; the explicit allowlist is the CI gate.

CI checks the publication boundary and scans all fetched Git history with a
checksum-verified secret scanner. GitHub Actions use pinned commits, read-only
repository permissions, and no persisted checkout credentials. Scan findings
must not be printed unredacted or broadly suppressed.

Secret scanning cannot determine whether ordinary-looking numbers came from a
real financial record. Human review of fixture provenance and public Actions
output remains necessary. A source audit is not a guarantee that every possible
software vulnerability has been found.
