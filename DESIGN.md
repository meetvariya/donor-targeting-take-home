# Donor Audience Service

## Decision And Scope

The objective is donation and dollar retention versus BAU, with fewer unsubscribes, not higher
AUC. This submission keeps the existing fitted pipeline and feature definition, adds an
authenticated durable API, and demonstrates delivery under enforced 2 GiB / four-CPU limits.
The routing agent and embedding model remain outside the service.

## Architecture And Timing

`POST /audiences -> SQLite job -> one bounded worker -> features/ranking -> one CRM import`.
Admission returns 202 and a status URL. `GET /requests/{request_id}` exposes completion, list ID,
count, versions, timings, or an explicit error. Success means the complete CSV and metadata are
already in the supplied CRM outbox, not merely that scoring finished.

After the nightly refresh, prepare an immutable serving generation, publish its pointer
atomically, and restart the one service instance before 09:00 ET. Preparation uses DuckDB with
a 512 MB operator budget, two threads and disk spill; it does not load the 12.7M-event timeline
into pandas. It retains sorted active IDs, 52 content embeddings, sparse distinct-content
engagements and their first timestamps. Arrays are checksummed and memory-mapped. A changed
source makes readiness/admission fail closed until preparation and restart; hot reload is
deliberately not implemented. First local startup can prepare missing artifacts.

At request time, trim engagements to `[requested_at - 30 days, requested_at)` within available
snapshot data, compute similarities, score the saved sklearn pipeline, select deterministically,
persist the audience and call the CRM. No new content needs a warehouse join: its embedding
comes from the request. Program area is validated metadata, not a same-area hard filter.

The starter uses the FIRST open/click per person/content, then applies the window. We preserve
that convention, including a later click not refreshing an old first-open timestamp. Duplicate
opens/clicks contribute once. Missing centroids remain NaN for fitted model preprocessing;
centroid-policy ranking puts them last, with person ID breaking ties. No preprocessing is refit
online. Tiny parity tests compare the compact score to the original and brute-force features.

For incidence row `w`, embeddings `E`, unit request vector `e` and `G = E E^T`, cosine is
`w(Ee) / sqrt(w G w^T)`. Averaging cancels from this ratio. Norms use 4,096-row batches of
person-by-content workspace, never a person-by-person or full person-by-1024 array. This is
exact centroid math, not an approximation using the 12 program areas.

Local Compose combines ingress and a worker in one process, mocking the deployment within the
hardware budget. A real Render background worker cannot expose HTTP. Production would need
separate ingress and an external durable queue/store feeding this worker; SQLite is for the
single-instance local implementation, not shared storage across Render services.

## Measured Resource Budget

The full recorded run is in `evidence/benchmark.json`; run `make benchmark` to regenerate it.
Docker configuration and cgroup accounting enforce 2,147,483,648 bytes and four CPUs.
The table records the first full run. Final rebuilt-image verification is retained separately
in `evidence/benchmark_final.json`: all 12 requests passed (maximum 2.48 s), the 250k import
took 2.67 s, and whole-container peak was 529.93 MiB. All 31 tests pass; one upstream
Starlette test-client deprecation warning remains.

| Measurement | Recorded result |
| --- | ---: |
| Real snapshot active audience | 244,057 |
| Compact arrays: 392,834 sparse edges plus IDs/embeddings | 7.49 MiB |
| Cold preparation of real warehouse | 2.77 s |
| Maximum receipt-to-CRM time over all 12 real requests | 2.46 s |
| Separate 250,000-member import, receipt to CRM completion | 2.49 s |
| Capacity case queue / scoring / CRM | 0.12 / 0.24 / 2.08 s |
| Whole-container peak during the complete benchmark | 534.73 MiB |
| OOM / OOM kill events | 0 / 0 |

The capacity fixture uses 250k unique synthetic active IDs with realistic sparse engagement
patterns. Its fraction=1 policy checks the maximum import payload; it is NOT the business
policy, and its warehouse/outbox are isolated from the evaluator. Every emitted list is checked
for uniqueness, eligibility, matching content metadata and counts. Cgroup peak includes the
API, benchmark client, cold preparation, capacity child and file cache, not just Python heap.

A dense float32 250k-by-1024 centroid array alone would take 976.6 MiB. The compact matrix is
7.49 MiB on real data; a 4,096-by-52 float64 norm workspace is about 1.63 MiB. Numeric scores,
IDs and selection arrays take single-digit MiB each. Python/native imports, DuckDB operators,
CRM list/dictionary/string copies and file cache dominate the measured peak. DuckDB's operator
budget alone is not a process-memory guarantee. There is substantial measured headroom, not
an assumption that all Python objects are free.

One worker limits memory duplication. Admission reserves 45 seconds per pending job and caps
pending work at three. A simultaneous 12-request burst admitted three, rejected nine with 429,
and completed every accepted job within its deadline. This is an explicit bounded workload,
not unlimited-concurrency assurance. Normal timing starts at HTTP receipt using a monotonic
clock; restart recovery uses persisted wall-clock deadlines. The given approximately two-second
CRM import is a dependency assumption; a real adapter needs bounded network timeouts.

## Audience Decision And Evidence

Historical random 75% sending supports replay on observed recipients. BAU is another 0.75
inclusion probability on that sample: normalize donor/dollar recall by 0.75 and avoided
unsubscribes as `1 - unsubscribe_share / 0.75`. Historical people must NOT be filtered using
their final snapshot status; that would introduce survivorship bias.

A model trained before July evaluated fractions 50%-80% on July. The 62.5% winner passed July
donor/dollar point-estimate guardrails but failed the frozen August check: 97.94% of BAU donors,
95.84% of dollars and 25.15% fewer unsubscribes. Model-ranked 75% retained more dollars but
slightly increased unsubscribes. Neither is described as meeting both business objectives.

The release uses the ORIGINAL centroid-top-75% safety reference, retaining the fitted model
for inference and the model-ranking alternative. It sends approximately 183k of the 244k active
people and suppresses the bottom quarter. Replaying our exact deterministic tie-break on all
60k historical sends estimates 107.25% of BAU donors, 107.21% of dollars and 1.23% fewer
unsubscribes. These differ from the starter's random-tie reference (108.0%, 108.3%, about 1.9%). August's
8,491 sends, six emails and 113 donors yield 102.65%, 102.21% and 11.11% respectively.
Adopting a pre-existing reference after a release gate is disclosed; its August numbers are
diagnostic, not independent confirmation of a newly selected policy.

The 12-request benchmark establishes delivery, deadline and output validity only. It does not
measure expected donations or churn for those new emails. Only the interviewer's unavailable
simulator can score those lists directly; our estimates come from randomized historical replay.
The full-history diagnostic includes development data and is not a separate holdout result.

Email-bootstrap August 95% ranges are 93.65%-111.11% for donors, 89.31%-112.73% for dollars and
4.58%-17.04% for unsubscribe reduction. This cannot establish zero revenue loss. Sparse
program-area slices sometimes lose dollars; one aggregate threshold is provisional, not a
promise for every area. The report retains the frontier, rejected candidate, references and
slices. If no tested policy passes the point-estimate gate, the approved fallback is BAU,
not sending everyone. Unknown or stale eligibility always fails closed.

## Persistence, Security And Failures

Persist the fitted model/card, versioned policy, serving arrays/manifest, SQLite requests and
payload hashes, deadline/status/result, selected audience/checksum and CRM output. Versions pin
each accepted job; changed artifacts cannot silently rescore recovered work. Equal request IDs
and payloads return the same job/result; a conflicting payload returns 409. Completed results
survive restarts. Queued/interrupted scoring recovers, while uncertain imports do not replay.

Persist `delivering` before the external call. A crash or timeout after possible CRM acceptance
becomes `delivery_unknown`; operator reconciliation is required. The supplied CRM has neither
a provider idempotency key nor a reliable correlation lookup. SQLite admission deduplication
does NOT guarantee exactly-once external delivery. Do not hide that failure window with retries.

Bearer authentication protects admission, status and readiness. The permitted org, audience and
destination are enforced. Inputs require 1024 finite values, nonzero norm, timezone-aware dates,
known program area and bounded fields/body. Validation errors omit payloads. Logs contain IDs,
counts, latency, versions and error types rather than recipient lists or embeddings. Compose
binds only localhost, drops capabilities and uses a read-only application filesystem. The
public demo token is strictly local; replace it for any deployment and provide TLS/auth at ingress.

Local jobs/audience artifacts older than seven days are purged on startup; the scheduled daily
restart enforces this retention and defines the idempotency horizon. Retain unresolved imports
for reconciliation. Mock CRM outbox retention is separate; production CRM retention/suppression
must be customer-controlled. Encrypt production state at rest and use tenant-scoped IAM before
multi-tenant deployment; a single demo-org token is not that system.

Demo mode uses the supplied September simulation timestamps, not today's October clock. Live
mode uses server time and rejects excessive clock skew or a stale snapshot; a real warehouse
adapter must supply its actual refresh watermark instead of this static snapshot constant.
Nightly status cannot observe same-day unsubscribe consent. Production must apply live CRM
suppression or a consent delta before sending; the provided mock cannot prove that protection.

## Live Validation And Next Work

First run shadow lists and reconcile counts/eligibility. Then use concurrent randomized BAU
controls with stable assignment, matured seven-day donation/dollar outcomes and unsubscribe
outcomes. Monitor p95/p99 receipt-to-CRM latency, queue depth/rejections, failed/unknown imports,
source age, missing-feature fraction and audience distribution. Stop on ineligible recipients,
unknown delivery, repeated SLA failures or breached revenue guardrails; roll back to the last
approved policy only when eligibility is valid. Otherwise reject requests.

Replay covers attributed seven-day gifts, not unattributed revenue, incremental causality or
long-term list retention under repeated targeting. Next priorities are live suppression,
provider-side idempotency, stronger temporal/program-area evidence and a longer randomized
retention experiment. Add a second worker, another model or distributed infrastructure only
after a measured bottleneck or a supported business benefit justifies it.