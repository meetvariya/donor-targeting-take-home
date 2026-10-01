# Copilot Planning And Implementation

## Scope

Used GitHub Copilot to inspect the brief, data dictionary, existing feature/model/baseline code
and tests; prioritize the productionization work; and implement the API, compact preparation,
policy evaluation, failure tests, container packaging and benchmark harness.

## Delegation And Checks

During planning, two read-only Explore reviews separately inspected serving constraints and
audience evaluation. Their claims were checked against the source. Implementation remained in
the main session, with focused tests immediately after each substantive edit.

Verified the original baseline, fitted AUC, numerical feature parity, temporal boundaries,
active eligibility, deterministic ranking, auth/validation, durable idempotency, queue recovery,
overload, corrupt/stale data and uncertain CRM delivery. The constrained Docker benchmark
replayed all 12 requests, tested a burst, prepared cold artifacts and imported 250k unique
synthetic active IDs in an isolated outbox. The JSON evidence is retained.

Final verification: 31 tests passed. The final rebuilt 2 GiB / four-CPU container completed
all 12 real requests within 2.48 seconds and the isolated 250k-member import in 2.67 seconds;
whole-container peak was about 530 MiB. A simultaneous burst admitted three and rejected
nine before acceptance. Dependency versions and source fingerprints are in the final record.

## Corrections And Overrides

- Rejected the suggestion that top-50% targeting preserves revenue. Correct BAU normalization
  shows a meaningful dollar loss in the supplied table.
- Did not accept estimated millisecond timings or memory numbers as measured evidence; used
  real end-to-end imports and cgroup memory.peak under Docker limits.
- Preserved the starter's first-engagement semantics rather than treating later clicks as a
  refreshed engagement window.
- A July-selected model cutoff failed August dollars. Kept that failure in the report and
  adopted the original fixed safety reference, explicitly without an independent-confirmation claim.
- Distinguished request admission deduplication from exactly-once CRM delivery. Ambiguous
  external success is surfaced for reconciliation instead of blindly retried.
- FastAPI initially interpreted a closure-local auth annotation as a query parameter; the
  focused tests identified it and the dependency declaration was corrected.
- The patch tool repeatedly changed indentation near a wrapped function signature. After
  three attempts, the user approved mechanical formatting; ultimately replacing the complete
  method fixed it. Tests were rerun before continuing.
- Non-finite input validation now returns sanitized errors rather than reflecting invalid
  float values or embeddings into JSON responses.

## Limits

No cloud deployment, live CRM suppression, causal revenue experiment or model overhaul was
claimed. No private secrets or personal details are included in this record.