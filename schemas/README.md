# VI pilot formats

Initial format version; no existing records migrated. Experiment parameters are
validated by `vi-experiment.schema.json`; registry records by
`vi-registry.schema.json`. `read_plan` also verifies references and the original
split. Run manifests and data manifests use `format_version: 1`; original
results and input files are immutable. Completed run directories cannot be reused.
