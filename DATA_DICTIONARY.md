# Data dictionary

The warehouse is a set of parquet files in `data/warehouse/`: `make data` downloads the tables
and `make features` adds `features.parquet`. The data is synthetic, and everyone works from the
same snapshot.
Query it with pandas, pyarrow, or DuckDB (`donor_targeting/mocks/warehouse.py` gives you a
DuckDB connection with one view per table).

**Refresh schedule.** The production warehouse is rebuilt every night at **01:00
America/New_York**. The files here are the refresh of **2026-09-01 01:00 ET** (`SNAPSHOT_AT` in
`donor_targeting/config.py`): they hold everything that happened before that moment and nothing
after it.

**Time zones.** Every timestamp is stored in UTC (`timestamp[ns, tz=UTC]`). Emails go out at
about 10:00 ET.

Row counts below are for the default full-scale build.

## `timeline`: one row per email event (12.7M rows)

| column | type | description |
|---|---|---|
| `id` | int64 | Event id, unique. Ascending in `occurred_at` order. |
| `content_id` | int64 | The email the event belongs to (`content.id`). |
| `person_id` | int64 | The subscriber (`people.id`). |
| `occurred_at` | timestamp (UTC) | When the event happened. |
| `type` | string | One of `sent`, `opened`, `clicked`, `bounced`, `unsubscribed`. |

- Every email from 2026-03-03 to 2026-08-27 was sent to a **random 75%** of the people who were
  active at send time. That is the business-as-usual (BAU) rule.
- Each `(person_id, content_id)` pair has at most one event of each type. `opened` and
  `clicked` come from tracking pixels and links. `clicked` implies `opened`.
- `bounced`: the address did not accept the email. A hard bounce (invalid address) also removes
  the person from the list (`people.status = 'bounced'`).
- `unsubscribed`: the person unsubscribed using that email's link and gets no further email.
  Between 1% and 2% of recipients unsubscribe from each email (1.5% on average).
- There is no `converted` type. A conversion is a row in `transactions`.

| rate per email (average) | value |
|---|---|
| sends per email | ~185,000 |
| opened / sent | 26.7% |
| clicked / opened | 13.9% |
| bounced / sent | 0.4% |
| unsubscribed / sent | 1.5% (range 1.1%–2.0%) |

## `transactions`: one row per donation (131k rows)

| column | type | description |
|---|---|---|
| `id` | int64 | Transaction id, unique. |
| `content_id` | int64, nullable | The email the donation is attributed to. NULL for donations not attributed to an email (web, events, direct mail): about 19% of donations. |
| `person_id` | int64 | The donor (`people.id`). |
| `amount` | float64 | Gift in USD. Median $45, mean $58, minimum $5. |
| `transaction_date` | timestamp (UTC) | When the gift was made. |

- **Attribution.** `content_id` is the email the payments platform credits the gift to (the
  donor arrived through that email's link or tracking). The data science team counts a send as
  **converted** when a transaction with the same `person_id` and `content_id` has a
  `transaction_date` in `[sent_at, sent_at + 7 days)`.
  `donor_targeting.warehouse_io.attributed_amount` implements this.
- About 1.1% of sends convert.

## `content`: one row per email (52 rows)

| column | type | description |
|---|---|---|
| `id` | int64 | Content id (1001–1052). |
| `date_sent` | date | The day the email went out (Tuesdays and Thursdays). |
| `program_area` | string | One of 12 program areas (see below). |
| `subject` | string | Subject line. |
| `dense_embedding` | fixed_size_list<float32>[1024] | L2-normalised embedding of the email's content (subject, body, imagery). |

Program areas (`donor_targeting.config.PROGRAM_AREAS`): `disaster_relief`, `hunger_relief`,
`clean_water`, `education`, `child_health`, `maternal_health`, `refugee_support`,
`climate_resilience`, `mental_health`, `housing`, `economic_empowerment`, `medical_research`.

- **Program area explains about 87% of the variance of `dense_embedding`.** Emails from the same
  program area sit close together. What is left is mostly tone and style (urgent appeal,
  matched gift, story, newsletter, research update, ...) plus noise.
- New content arrives through the chatbot. The routing agent embeds it with the same embedding
  model and classifies its `program_area` with an LLM (see `examples/requests/`).

## `people`: one row per subscriber (392k rows)

Not required by the brief, but you will need it to know who can be emailed.

| column | type | description |
|---|---|---|
| `id` | int64 | Person id. |
| `subscribed_at` | timestamp (UTC) | When they joined the list. |
| `status` | string | `active` (244k), `unsubscribed` (145k) or `bounced` (3k) as of the snapshot. |
| `status_updated_at` | timestamp (UTC), nullable | When they left the list. NULL while active. |

Only `active` people may be emailed. About 5,000 people join the list each week.

## `features`: the data science team's training features (60k rows)

Written by `make features` (`donor_targeting/features.py`) for a **random sample** of 60,000
historical sends. It only covers sends with a full 30-day look-back and a fully observed 7-day
attribution window: 2026-03-31 to 2026-08-20.

| column | type | description |
|---|---|---|
| `timeline_id` | int64 | The `sent` event (`timeline.id`) the features describe. |
| `centroid_1m` | list<float32> (1024 values), nullable | The mean `dense_embedding` of the distinct content the person opened or clicked in the 30 days before the send. NULL when they engaged with nothing (about 25% of sends). |
| `centroid_cosine_similarity` | float32, nullable | Cosine similarity between `centroid_1m` and the sent content's `dense_embedding`. NULL when `centroid_1m` is NULL. |

`donor_targeting.warehouse_io.modelling_frame()` joins these features to their send and
outcomes (`amount`, `donated`, `unsubscribed`).

## Outside the warehouse

| path | what |
|---|---|
| `data/requests/*.json` | 12 upcoming chatbot requests (content ids 2001–2012, one per program area), as the routing agent sends them. |
