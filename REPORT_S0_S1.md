# S0 and S1 Report Section

## S0: Prompt-Only Baseline

`S0` is the base translation system built directly on `Qwen2.5-7B-Instruct` without any task-specific adaptation. The model receives only the translation prompt and the English source sentence, and it generates a Japanese translation using the shared decoding configuration used across variants. This baseline is important because it shows how far a strong instruction-tuned model can go with prompting alone before any fine-tuning or retrieval is introduced.

In our implementation, `S0` acts as the reference point for the rest of the system family. Any improvement in later variants should therefore be interpreted relative to `S0`, not in isolation.

## S1: Fine-Tuned Model with LoRA Adapters

`S1` extends `S0` by adding QLoRA fine-tuning on top of the same base model. The goal of this stage was to adapt the model more closely to the English-to-Japanese translation task and reduce common failures such as unnatural phrasing, prompt leakage, and inconsistent terminology.

The current fine-tuning configuration is:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | QLoRA |
| Adapter output | `translation_v2/final` |
| Training rows | 24,000 |
| Dev rows | 2,668 |
| Epochs | 3 |
| Learning rate | `2e-5` |
| LoRA rank | 64 |
| LoRA alpha | 64 |
| Batch size | 8 |
| Max sequence length | 2048 |
| Eval interval | Every 200 steps |
| Early stopping patience | 2 |

During inference, `S1` uses the same decoding configuration as `S0` and differs only by loading the fine-tuned adapter. This makes the comparison between `S0` and `S1` a direct measure of fine-tuning gain.

## Evaluation Setup

Both `S0` and `S1` were evaluated on the same held-out English-to-Japanese test set containing 4,254 samples. As of April 6, 2026, `test_v2` is still empty in the local workspace, so the current comparison uses `data/splits/test_v1.jsonl` as the common evaluation set for both variants.

We report:

- `BLEU` and `chrF++` as surface-form overlap metrics against the reference translation
- `COMET` as the main learned quality metric
- `Terminology Accuracy` as a separate glossary-compliance metric
- `Average Latency` to capture runtime cost

In addition to aggregate metrics, we also ran a pairwise comparison of every `S0` and `S1` prediction to measure how often the fine-tuned model actually changes the output, improves it, or makes it worse.

## Quantitative Results

| Metric | S0 | S1 | Delta (S1 - S0) |
| --- | ---: | ---: | ---: |
| Num Samples | 4254 | 4254 | 0 |
| Avg Latency (ms) | 1122.6 | 1259.4 | +136.8 |
| BLEU | 5.74 | 5.66 | -0.08 |
| chrF++ | 21.49 | 21.48 | -0.01 |
| COMET | 0.8437 | 0.8450 | +0.0013 |
| Terminology Accuracy | 0.5787 | 0.5751 | -0.0036 |
| Terminology Terms Total | 5943 | 5943 | 0 |
| Terminology Correct Terms | 3439 | 3418 | -21 |

### Pairwise Output Comparison

| Comparison Signal | Value |
| --- | ---: |
| Identical `S0` and `S1` outputs | 2950 / 4254 |
| Identical output rate | 69.35% |
| Rows where `S1` improved | 631 |
| Rows where `S1` worsened | 564 |
| Rows with no meaningful quality change | 3059 |
| Contaminated outputs in `S0` | 1381 |
| Contaminated outputs in `S1` | 1352 |
| Bad `S0` outputs cleaned up by `S1` | 71 |
| Bad `S1` outputs where `S0` was cleaner | 42 |

## Interpretation

The overall result is that the current `S1` fine-tuning stage is largely flat relative to `S0`, rather than a strong improvement. The aggregate scores are almost unchanged: `COMET` increases slightly, but `BLEU`, `chrF++`, and terminology accuracy all decline marginally. This means the fine-tuned adapter does not yet provide a clear quality gain on the held-out evaluation set.

However, the pairwise comparison reveals a more nuanced picture. `S1` is not simply identical to `S0`: it improves 631 rows and worsens 564 rows, while leaving most outputs unchanged. In practice, the main benefit of `S1` is that it repairs some pathological baseline failures, especially outputs that contain English fragments, meta-explanations, or mixed-language contamination. For example, several `S0` outputs contained explicit explanatory text such as “This translation preserves the original meaning and tone...” or untranslated English phrases, while `S1` often converted these into a clean Japanese sentence.

At the same time, `S1` still produces many contaminated outputs, and in some cases it introduces new prompt leakage or formatting artifacts that were not present in `S0`. This explains why the aggregate quality improvement is so small: the adapter appears to reduce some severe errors, but not consistently enough to shift the overall system into a clearly better regime.

## Qualitative Summary

Representative `S1` improvements include:

- turning a baseline meta-answer into a direct translation, such as translating “The yacht was sabotaged.” from an explanatory paragraph in `S0` into a concise Japanese sentence in `S1`
- removing mixed-language noise in examples where `S0` contained English, romanized Japanese, or Chinese fragments
- producing more natural sentence-level Japanese in some short conversational examples

Representative `S1` regressions include:

- reintroducing English explanation text inside the answer
- leaving technical terms or phrases partially untranslated
- changing a semantically correct `S0` sentence into a less accurate paraphrase

These examples suggest that the current fine-tuned adapter has learned some translation-specific cleanup behavior, but has not yet learned stable output discipline.

## Conclusion for S0 vs S1

For the `S0` vs `S1` comparison, the fine-tuning gain is limited. `S1` is best described as a modest stabilisation step rather than a major performance improvement. It slightly improves learned quality according to `COMET` and reduces some obviously bad baseline outputs, but it does not materially improve overall translation quality on the current held-out set and incurs higher latency.

This makes `S1` a useful intermediate system, but not yet a convincing final improvement over the base model. The next iteration should focus on reducing output contamination, aligning training and test splits more cleanly, and verifying whether the adapter is learning translation behavior or simply memorising noisy response patterns from the training data.
