# Direct-plus-hurdle ensemble registration

This is a new prospective confirmation, not a continuation or rescue of the failed pilot registration.

The exact frozen predictor is `0.5 × direct + 0.5 × hurdle` on the probability scale. Both candidate-specific direct and hurdle models, all preprocessing, and the common h10 comparator come from the checksum-sealed pilot. No fitting or recalibration exists in the confirmation path.

Registration ID: `c75965bd6c5beb46e4bf53b453f01cdd14a0a3c8e8752bc50e2d562139976935`
Source pilot seal: `4db89c5095682c4cf055ed0cb26f9ba80972fd849e8f9a722fe6edf84b3b08a7` (`stopped_before_confirmation`)
New matrices: **200**
New primary futures: **256,000**
Full exact replay: **required**.

The motivating pilot ensemble result is explicitly post-hoc and developmental:

- Candidate-equal out-of-fold loss: `0.110625970753`.
- Candidate 02 half A developmental gain: `0.006403179292`.
- Candidate 02 half B developmental gain: `0.005324132165`.
- Candidate 03 half A developmental gain: `0.008693537552`.
- Candidate 03 half B developmental gain: `0.009464369257`.

The prospective claim succeeds only if the ensemble beats h10 in all four candidate-by-half cells, with a positive gain, a positive whole-matrix bootstrap lower bound, and Holm-adjusted whole-matrix randomization `p < 0.05` in every cell.

Constituent, Brier, rank, and incidence results are secondary and cannot rescue failure.
