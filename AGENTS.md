# AGENTS.md — Research and Coding Agent Contract

## Mission

This repository investigates **real-data-driven calibration of simulators for deformable-object manipulation** and whether such calibration improves **sim-to-real transfer of robotic policies**.

The primary outcome is **measured real-world task performance**, not a visually convincing simulation or a low calibration loss alone. Treat negative results, failed replications, and limitations as valuable scientific evidence.

> Repository status: the README describes a target architecture. Do not assume that a documented CLI, adapter, dataset, benchmark, run, or experiment already exists. Inspect the filesystem before using it.

## 1. Scope and priorities

Work within the following research loop:

1. State a falsifiable hypothesis.
2. Specify an experiment and an evaluation protocol **before** interpreting results.
3. Collect or inspect traceable real-world observations.
4. Estimate simulator parameters and uncertainty on calibration data.
5. Evaluate dynamics fidelity on held-out data.
6. Train or evaluate policies under controlled conditions.
7. Evaluate sim-to-real transfer using an independent real-world protocol.
8. Record evidence, uncertainty, and findings, including contradictory evidence.

Favor **reproducibility and valid comparisons** over higher headline numbers.

## 2. Source-of-truth boundaries

| Location | Meaning | Agent behavior |
|---|---|---|
| `research/` | Human-readable scientific knowledge and Foam graph | Create well-linked Markdown; clearly separate proposals from evidence |
| `registry/` | Baseline, benchmark, and protocol technical specs | Make versioned, schema-valid changes |
| `experiments/` | Executable experiment plans | Use explicit IDs, configurations, and constraints |
| `src/lab/` | First-party implementation | Keep modular contracts, tests, and clear dependencies |
| `third_party/` | External code, often Git submodules | Do not edit upstream code in place; use wrappers, forks, or explicit patches |
| `datasets/` | Data manifests, schemas, and references | Never silently change or relabel raw measurements |
| `runs/` | Execution manifests, metrics, and artifact references | Append new run records; never rewrite raw results to fit a conclusion |
| `schemas/` | Validation contracts | Update schemas and migration notes when formats change |

**Do not duplicate authoritative configuration fields across Markdown and YAML.** YAML plans/registry files own executable parameters. Markdown owns motivation, interpretation, and narrative. Wikilinks may be generated from authoritative metadata if tooling exists.

## 3. Research entity conventions

Use stable IDs and corresponding Markdown filenames:

- `Q-0001` — research question (`research/questions/Q-0001.md`)
- `H-0001` — hypothesis (`research/hypotheses/H-0001.md`)
- `E-0001` — experiment (`research/experiments/E-0001.md` and `experiments/E-0001.yaml`)
- `BL-0001` — baseline (`research/baselines/BL-0001.md` and `registry/baselines/BL-0001.yaml`)
- `B-0001` — benchmark (`research/benchmarks/B-0001.md` and `registry/benchmarks/B-0001.yaml`)
- `P-0001` — protocol (`research/protocols/P-0001.md` and `registry/protocols/P-0001.yaml`)
- `R-0001` — run (`runs/R-0001/manifest.json` when executed)
- `RR-0001` — run review (`research/run-reviews/RR-0001.md`, referencing an executed run)
- `F-0001` — finding (`research/findings/F-0001.md`)

Markdown notes must contain YAML frontmatter with at least `id`, `type`, and `status`; use `[[H-0001]]`-style wikilinks for Foam navigation. Do not use the same ID for different entities.

Use explicit lifecycle states; do not promote a record to `completed`, `supported`, or `reproduced` without corresponding evidence. A finding must cite **specific run IDs and protocols**, not only another agent's summary.

## 4. Experimental rigor

### Required experiment plan

Before running a scientific experiment, specify:

- Research question, hypothesis, falsification or decision criteria.
- Object category and instance IDs (cloth, rope, soft body, etc.), robot, sensors, task and environment.
- Baselines and the **versioned** benchmark/evaluation protocol.
- Calibration, validation, held-out test, and real-world evaluation splits.
- Simulator, model class, optimized parameters, priors/bounds and optimizer.
- Primary and secondary metrics, units, aggregation method, uncertainty method.
- Seeds, number of trials, sample/data budget, compute budget and stopping rules.
- Safety constraints and human approvals for real hardware.

If a required item is unknown, record `TBD` and mark the experiment as a **proposal**. Never manufacture missing protocol details.

### Avoid leakage and unfair comparisons

- Split related frames by trajectory/session/object/condition as required by the protocol; adjacent frames are not independent samples.
- Never fit physical parameters on held-out test episodes, then cite those episodes as independent validation.
- Do not tune policies on the final real-world evaluation set.
- Compare calibration methods with matched policy architecture, training budget, dataset access, seeds, and evaluation conditions where feasible. Document every exception.
- Separate effects of simulator calibration from effects of policy changes or different data budgets.
- Treat literature-reported results as `published`, not `reproduced`.
- Report failed episodes and negative findings; do not cherry-pick successful trajectories.

### Deformable-object modeling

- Specify parameter units and semantics (stiffness, damping, friction, mass/density, geometry, contact, plasticity only when supported).
- Record coordinate frames, axis conventions, camera calibration, timestamps, clock synchronization, action/control rate, and calibration residuals.
- Make uncertainty and potential non-identifiability visible; do not present fitted parameters as unique physical truth without evidence.
- Keep observation loss / simulation fidelity separate from downstream real-world policy success.
- Do not assume that all simulators expose interchangeable deformable-body physics.

## 5. Baseline and benchmark integration

Expose first-party interfaces; isolate external implementations behind adapters:

- **Policy adapter:** load checkpoint, preprocess observations, emit actions, reset state.
- **Simulator adapter:** configure parameters, reset, step, export observations/state.
- **Benchmark adapter:** enumerate tasks, construct episodes, and evaluate outcomes.
- **Protocol:** defines splits, seeds, repetitions, metrics, stopping criteria and aggregation.

Reject mismatched sensors, control frequencies, coordinate frames, action units or shapes rather than performing silent coercion.

Prefer a pinned package when available; use a pinned Git submodule when exact upstream source is required; isolate conflicting runtimes in separate containers/processes. Never modify `third_party/` without an explicit, documented fork or patch path. Record upstream commit hashes and local patches in provenance.

A minimum meaningful comparison should include an **uncalibrated simulator** and the **real-data-calibrated simulator**; include nominal/manual calibration and domain randomization when feasible. Do not label an unexecuted method as a measured baseline.

## 6. Data and provenance requirements

Raw observations are immutable. Changes to alignment, denoising, segmentation, reconstruction, or coordinate transforms produce **derived datasets** with distinct versions and processing manifests.

Before interpreting data, inspect/record:

- Object instance and session IDs, acquisition conditions, sensor modalities and valid timestamps.
- Units and transforms across camera, robot, world and simulation frames.
- Robot control mode and action semantics; available force/tactile signals.
- Missing observations, sensor dropouts, confidence masks and measurement uncertainty.
- Calibration procedure and identifiers for training/validation/test partitions.
- Privacy, consent, access restrictions, and dataset license where relevant.

All executed runs must retain a machine-readable manifest containing experiment/baseline/benchmark/protocol IDs, repository and external commits, config hashes, dataset/checkpoint hashes, environment versions, seeds, hardware details when relevant, metrics, logs/artifact locations, status, and any failure reason.

Store large media and checkpoints in versioned artifact storage when possible; use checksummed references in Git. Do not commit secrets, credentials, private images, or sensitive sensor recordings without permission.

## 7. Code changes and validation

- Prefer small, testable, composable Python modules over one-off scripts.
- Preserve backwards compatibility of schemas/IDs where possible; document intentional migrations.
- Use typed interfaces and deterministic seeds where applicable; never assume deterministic GPU physics without verification.
- Add unit tests for geometry/coordinate conversions, metric calculations, split logic, adapter compatibility and schema validation.
- Use smoke tests for adapters; scientific or hardware evaluations must not run automatically as part of ordinary pull-request CI.
- Preserve error messages and debug artifacts; do not swallow failures to produce a successful report.
- Avoid speculative dependencies or claiming that an unimplemented command works.

Before finishing, run **only checks supported by the actual repository**. Report exactly what you ran and what you could not verify. Never fabricate test, calibration, benchmark, training or hardware results.

## 8. Hardware and safety boundaries

**No autonomous real-world robot execution without explicit human authorization for the specific task and setup.**

- Default to read-only analysis, offline data processing, code inspection, tests, and simulation.
- Before any physical experiment, require an approved safety protocol, suitable supervision, robot workspace limits, collision/force constraints, accessible emergency stop and confirmation that the scene is safe.
- Never silently change hardware limits, contact thresholds, control rates, or emergency-stop settings.
- Do not control a physical robot merely because an API or ROS topic is accessible.
- Distinguish simulated outcomes from physical measurements in every report.

## 9. Agent working procedure

For each task:

1. **Inspect:** read relevant README, research notes, registry specs, plans, schemas, existing code and run manifests.
2. **Scope:** state what is known versus uncertain; identify which hypothesis or capability the change serves.
3. **Plan:** prefer the smallest reversible change; identify required schema/metadata changes.
4. **Implement:** keep third-party code untouched and add interfaces/tests where needed.
5. **Validate:** run local supported checks; avoid costly or physical execution unless authorized.
6. **Record:** update relevant Markdown links and machine-readable specs; create a finding only with actual evidence.
7. **Report:** summarize files changed, checks performed, data used, limitations, and next steps.

When producing research interpretations, label statements as one of:

- **Observation** — directly measured and traceable to a run/dataset.
- **Inference** — interpretation supported by specified evidence.
- **Hypothesis** — untested or under-tested proposition.
- **Proposal** — future experiment or implementation idea.

If evidence is inconclusive, say so. Prefer to propose a discriminating experiment over asserting an unsupported conclusion.

## 10. Definition of done

A research/code change is complete only when all applicable items are satisfied:

- [ ] The change serves a documented research question/hypothesis or infrastructure need.
- [ ] IDs, wikilinks, YAML plans and registries remain consistent.
- [ ] Reproducibility metadata and data provenance are preserved.
- [ ] No calibration/test leakage or misleading baseline comparison is introduced.
- [ ] Relevant tests/validators pass, or unverified checks are explicitly disclosed.
- [ ] No raw data, immutable results, third-party source or safety settings are silently overwritten.
- [ ] Human approval is obtained before any real-world robot operation.
- [ ] Claims about improvements cite actual runs and their evaluation protocols.

## 11. Preferred communication

Write code, field names, stable IDs and machine-readable schemas in **English**. Use clear scientific prose in research notes; follow the language of the existing document or user request. Be concise, explicit about unknowns, and never imply a planned component or experiment is already implemented.
