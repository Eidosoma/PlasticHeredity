# Chapter 5 checkpoint archive manifest

On 2026-08-19, before the Phi-r extension program generated scientific data,
the old ignored Chapter 5 checkpoint trees were archived to:

`/mnt/bioIce1/PlasticHeredityArchivedWorkfiles/ch5_pre_extension_2026-08-19/`

Each archive passed `gzip -t`, a complete `tar -tzf` traversal, and SHA-256
verification before its local checkpoint tree was removed. Final scientific
results and reports were not removed.

| Archive | SHA-256 |
| --- | --- |
| `phir_ch5_pilot_work.tar.gz` | `331deb9556a32b89a12bd658d9cf2c68b65909c2abfbc9e1666979ebf97b866e` |
| `phir_ch5_pilot_work_failed_pre_amendment_001.tar.gz` | `b5145ce97041bc4a05d68409ee9e4b2aa78b4311a529d8766fd88f3a06fafb9b` |
| `phir_feedback_dose24_work.tar.gz` | `55bd834e8ee5ba1e7a45057e6387b4071be77b0959aa64e1a1eab41c4919ceea` |
| `phir_protocol_adjudication_work.tar.gz` | `3e94eb849c13ab1d0fe6d1b5c1195c5f1023230a3454392d7f33415face6de55` |
| `phir_rescue_r0_work.tar.gz` | `fd87676d8625b52fd9413def27c6e76ba273d9a0556a6986cd34182ccab0ed4e` |
| `phir_rescue_r0_work_failed_pre_amendment_001.tar.gz` | `402ca25bc09e9a28b07b7c25751240be51696e33fae3131a3653d0867c911c2e` |
| `phir_window_bridge24_work.tar.gz` | `a9f639890f9084e306f7ca65c4a362f6d43e545b44733b228c676def1d324dad` |

The external directory also contains its authoritative `SHA256SUMS` file.

## Phi-r extension archives

After PX1 completed and passed exact replay/readback, its ignored checkpoint
tree was archived separately and then removed locally:

| Archive | SHA-256 |
| --- | --- |
| `/mnt/bioIce1/PlasticHeredityArchivedWorkfiles/ch5_extension_2026-08-20/px1_work.tar.gz` | `bd2686768cdbc6af8588288c58b0160b8ce832cb59fb5b672994a0497f8abeee` |

The archive passed gzip, tar traversal, and SHA-256 verification before the
local checkpoint copy was removed. The sealed PX1 results remain in place.
