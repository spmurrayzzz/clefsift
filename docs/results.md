# Recorded results and limits

Training and export completed on 2026-10-06 UTC.
These figures describe internal English validation with provisional labels.
They do not establish LinkedIn accuracy or verified human authorship.
The [model card](https://huggingface.co/spmurrayzzz/clefsift) records the published model details.

## Corpus and protocol

The source was `iitolstykh/LLMTrace_detection` at revision `332053804f8798177ff2ecb1148e1839ed429ba3`.
Its English inventory contained 40,317 rows in 15,459 source groups.
The structural audit found invalid mixed spans, missing human originals, and cross-split duplicate candidates.
Selection excluded known duplicate groups and malformed or high-Cyrillic candidates.
Authorship and source rights remain unresolved despite the upstream Apache-2.0 declaration.

v1 used 300 training groups and 50 validation groups, with human, AI, and mixed targets.
v2 retained 298 training groups and 47 validation groups, with four targets per group.
Complete requests fit 1,536 tokens without truncation.
Official source splits remained intact.

Both runs started from Clef revision `2f3de3dd85f379784083b0814d997ab627200f0c`.
Each used seed 0, rank 64, and 128 steps with 512 example exposures.
No synthetic bootstrap examples or commercial detector outputs supplied training labels.

v2 used `gpt-5.6-luna` for development edits and insertions and `gpt-6-luna` for held-out variants.
Training mixed targets comprised 124 Luna insertions and 174 upstream Gemini 2.5 Flash examples.
The two validation panels share 47 parent groups and their human/fully-AI controls.
Only the AI-assisted and mixed variants differ.
The held-out check tests another Luna version, not an unrelated model family.

The final test entered structural and duplicate auditing only.
It did not enter generation, prompt development, training, selection, or scoring.
The public tools do not import it.

## Reference results

Step 128 won matched checkpoint selection before held-out scoring.
F1 uses fixed target sets, with all four prediction options retained.

| Validation panel | Untuned accuracy | v1 accuracy | v2 accuracy | Untuned F1 | v1 F1 | v2 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original, 150 requests | 56.0% | 85.3% | 85.3% | 0.5243 | 0.8552 | 0.8766 |
| Matched v2, 188 requests | 46.3% | 64.9% | 89.9% | 0.4030 | 0.5585 | 0.8991 |
| Held-out Luna, 188 requests | 47.3% | 67.0% | 86.2% | 0.4276 | 0.5813 | 0.8602 |

Original validation uses fixed-three F1. The v2 panels use fixed-four F1.
v2 correctly classified 169/188 matched, 162/188 held-out, and 128/150 original validation requests.

Human false positives fell from v1's 6/47 to v2's 5/47 on each new panel.
A one-case difference is inconclusive.
AI-assisted recall was 45/47 matched and 33/47 held-out.
On original validation, mixed recall fell from v1's 45/50 to v2's 39/50, or 90% to 78%.

v2 added AI-assisted examples, mixed insertions, and text normalization together.
No ablation separates their effects.
One seed does not establish training stability.

## Selected reference and Q8

| Panel | Reference correct | Q8 correct | Reference F1 | Q8 F1 | Reference Brier | Q8 Brier |
|---|---:|---:|---:|---:|---:|---:|
| Matched | 169/188 | 168/188 | 0.8991 | 0.8936 | 0.1829 | 0.2737 |
| Held-out Luna | 162/188 | 162/188 | 0.8602 | 0.8602 | 0.2421 | 0.3115 |
| Original validation | 128/150 | 128/150 | 0.8766 | 0.8766 | 0.2596 | 0.3374 |

Brier sums squared error over four probabilities. Lower is better.
The original row still uses fixed-three F1.

Q8 retained 525/526 reference class predictions.
One matched AI prediction became mixed.
Human false positives and AI-assisted recall did not change.
Four display verdicts became uncertain under unchanged `0.35` and `0.10` thresholds.

Q8 probabilities were softer. The comparison includes merging, conversion, quantization, and backend differences.
It does not isolate quantization alone. No intermediate F16 evaluation ran.
The panels share controls and groups, so these are not 526 independent observations.

The retained Q8 file occupies 26.76 GiB.
Its SHA-256 is `1fbdedec7989395973e5364a47dcd4a01a5f419a425b26ba476faed5556cb7fa`.
The FP32 trained head remains part of the reference export.
Fine-tuned vision behavior is untested.

## Open evaluation work

- Independent professional posts with verified authorship and usage rights.
- The original mixed-recall regression and held-out AI-assisted recall gap.
- Unrelated generator families, multiple seeds, and separate component tests.
- Independent calibration and uncertainty analysis before threshold changes.

Use ClefSift for exploratory review with human judgment.
Do not use its output as the sole basis for academic, employment, or moderation decisions.
