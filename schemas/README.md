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

E-0006 adds `posterior.family: logistic_normal_stiffness` with an explicit
`uniform_stiffness` prior, positive lower/upper support and quadrature order.
`initial_std` is now family-specific: log stiffness for the existing Gaussian,
unconstrained logit for the new family. New records explicitly label this coordinate.
The new point estimate is median Y, and width ratios use physical Y standard
deviation relative to uniform standard deviation. Old checkpoints remain unchanged;
loading uses their stored posterior family. Optional `checkpoint_every` defaults to
1 and controls rolling checkpoint I/O; milestone and final checkpoints are retained.

E-0011 adds `low_rank_logistic_normal_stiffness`, requiring positive integer
`rank` and the existing uniform support fields. Covariance is diagonal plus a
low-rank factor in latent coordinates; physical marginals remain bounded.
Old family names and checkpoints retain their interpretation. Optional
`tensorboard.enabled` and `tensorboard.image_every` control online diagnostics.

## Graph residual posterior (additive)

`vi-experiment.schema.json` accepts `graph_residual_flow` and requires its graph,
network and stochastic trace settings. Existing family formats are unchanged.
The runtime additionally requires trace continuation probability greater than
the squared contraction bound. No existing records need migration.
