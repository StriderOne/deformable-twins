# VI pilot formats

Initial format version; no existing records migrated. Experiment parameters are
validated by `vi-experiment.schema.json`; registry records by
`vi-registry.schema.json`. `read_plan` also verifies references and the original
split. Run manifests and data manifests use `format_version: 1`; original
results and input files are immutable. Completed run directories cannot be reused.

E-0005 adds optional `posterior.prior_center_factor`, `variants` (only prior
center/scale overrides), and `checkpoint_iterations` to experiment format v1.
Existing plans retain factor 1 and the previous checkpoint behavior; no migration
is required. Variant plans require explicit CLI `--variant`; manifests retain the
selected variant and resolved posterior settings as well as the source config hash.
