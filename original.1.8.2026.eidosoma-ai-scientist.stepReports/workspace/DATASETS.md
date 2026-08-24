# Datasets

No dataset inputs are required for this experiment.

DATASET_CATALOG.json and DATASET_AVAILABILITY.json are still materialized for auditability. You can ignore them unless the user asks you to inspect available datasets.

Path rules: `/datasets` is read-only mounted data when mounts exist, `/cache` is disposable cache, and `$ARTIFACTS_DIR` is for collectible reports, manifests, logs, summaries, and small validation outputs.
