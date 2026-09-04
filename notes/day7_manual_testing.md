# Day 7 — Manual API Testing

How this was actually run: the task calls for testing through /docs (Swagger
UI) by hand. I don't have a browser to click through, so I made the
equivalent HTTP requests programmatically (same endpoints, same JSON bodies
Swagger UI would send) against a real running `uvicorn app.main:app`
process, rather than through the UI itself. Worth doing the UI version too —
it's a genuinely different, faster way to build intuition than reading a
table of results, and takes a few minutes.

Detailed per-customer results: [`day7_manual_score_notes.csv`](day7_manual_score_notes.csv).

## 20 individual customers via /score

15/20 matched my hand-reasoned "expected" tier going in. The 5 mismatches
were the useful part — each is a real finding about the model, not noise
(details and reasoning in the CSV's `why` column):

- **`7590-VHVEG`** (expected high, scored low, 0.223): month-to-month +
  electronic check + tenure=1 *looks* like the canonical high-risk profile,
  but this customer has `InternetService=DSL` and low `MonthlyCharges`
  ($29.85). `InternetService_Fiber optic` is the model's #2 feature by
  importance (Day 3, 0.320 — nearly as strong as `contract_risk`'s 0.372),
  and this customer doesn't have fiber. My heuristic only accounted for
  contract + payment method, not internet type. The model's call matches
  ground truth (this customer didn't churn).
- **`5575-GNVDE`, `7795-CFOCW`, `6827-IEAUQ`, `8627-ZYGSZ`** (all expected
  medium for "one-year contract," all scored low, 0.03–0.07): my assumption
  that a one-year contract should land in the medium tier was wrong across
  every example tried. `contract_risk` ordinally encodes one-year as 1
  (between month-to-month=2 and two-year=0), but the model's *actual*
  learned risk for one-year customers sits much closer to two-year's than
  to month-to-month's — ordinal position in the encoding doesn't mean
  linear risk in the model's output. Worth remembering when eyeballing any
  customer on a one-year contract: default assumption should be low, not
  medium, unless something else about them stands out.

## tenure == 0 customers specifically

Tested 5 of the dataset's 11 `tenure == 0` customers directly (all
`Two year` contracts, all `Churn=No` in the ground truth). All 5 scored
confidently low (0.003–0.013), no errors, no NaNs. This is the edge case
that broke things twice earlier in the project (the `TotalCharges`/
`charge_trend` NaN handling in Day 2, and the single-customer one-hot bug
found and fixed in Day 4) — good to see it's clean end-to-end through the
actual API now, not just in the notebooks.

## 100-customer batch via /batch-score

- **NaNs**: zero. Checked `churn_probability.isna().sum()` and an
  out-of-[0,1]-range check explicitly — both zero.
- **Speed**: 100 customers, 0.249s wall-clock round-trip including HTTP
  overhead (server-side logged latency was lower, see
  `app/main.py`'s `batch_score_request` log line) — not slow at this size.
  Worth re-checking this note once real customer volume (~7,000) and a
  real network hop are involved, rather than assuming this number holds at
  70x the size on localhost.
- **Tier boundaries**: spot-checked the rows nearest the 0.7 and 0.4
  cutoffs. `0.7139 → high`, `0.6821 → medium` (correctly split on either
  side of 0.7); `0.4044 → medium`, `0.3826 → low` (correctly split on
  either side of 0.4). Boundaries are inclusive on the high side
  (`>= threshold`), confirmed both in this run and by the unit tests on
  `assign_risk_tier` in `notebooks/04_scoring_pipeline.ipynb`.
- Tier distribution on this batch: 59 low / 12 medium / 29 high — broadly
  consistent with every other sample pulled from this dataset so far (Day
  4's notebooks, the earlier 300-customer batch test), nothing here
  suggests drift.

## Using this as a regression test set later

`day7_manual_score_notes.csv`'s `customer_id`, `expected_tier`, and
`actual_tier` columns are exactly the shape a regression test would check
going forward: re-run these same 20 customer_ids through /score after any
future retrain or feature-pipeline change, and diff `actual_tier` against
what's recorded here. A tier flip on any of these without a corresponding,
explainable reason (a real retrain, a real data change) is the signal
something broke — the same role this file's `why` column already plays for
today's mismatches, applied forward instead of backward.
