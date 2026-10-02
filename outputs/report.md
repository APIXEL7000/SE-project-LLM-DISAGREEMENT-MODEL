# Disagreement Analysis Report

> **SIMULATED RUN.** The mock backend is a simulator, not a language model. These numbers validate the pipeline only and are **not** evidence about real LLMs.

## Run configuration
- backend: `mock`  |  models: model-A, model-B, model-C
- samples per (prompt, model): 5  |  temperature: 1.0
- flag item when agreement < 0.75

## Summary
```json
{
  "n_items": 26,
  "n_flagged": 10,
  "pct_flagged": 38.5,
  "mean_agreement": 0.8231,
  "mean_entropy_norm": 0.3509,
  "mean_invalid_rate": 0.0,
  "n_cross_model_disagree": 3,
  "agreement_threshold": 0.75,
  "human_comparison": {
    "n_items_with_human_votes": 20,
    "mean_js_divergence": 0.0222,
    "mean_total_variation": 0.0867,
    "majority_match_rate": 0.95,
    "n_overconfident": 0,
    "pearson_llm_vs_human_entropy": 0.8813420018051273,
    "by_human_agreement": {
      "high_human_agreement": {
        "n": 17,
        "mean_llm_agreement": 0.8549,
        "majority_match_rate": 1.0
      },
      "low_human_agreement": {
        "n": 3,
        "mean_llm_agreement": 0.5778,
        "majority_match_rate": 0.6667
      }
    }
  }
}
```

## Uncertain cases (top 10 of 10 flagged, lowest agreement first)
| item_id | text | majority_answer | agreement | entropy_norm | cross_model_disagree | human_agreement | overconfident |
|---|---|---|---|---|---|---|---|
| nli05 | Premise: A child is holding a red balloon. Hypothesis: The child is at... | neutral | 0.5333 | 0.6289 | True | 0.6 | False |
| nli06 | Premise: Most students passed the exam. Hypothesis: Every student pass... | contradiction | 0.5333 | 0.6289 | True | 0.5 | False |
| sar04 | Nice weather we're having. | not_sarcastic | 0.5333 | 0.9968 | True | 0.5 | False |
| nli07 | Premise: The meeting was moved to Friday. Hypothesis: The meeting will... | neutral | 0.6667 | 0.5794 | False | 0.5 | False |
| nli14 | Premise: Nobody was injured in the accident. Hypothesis: The accident ... | neutral | 0.6667 | 0.7312 | False | 0.6 | False |
| nli13 | Premise: The study found a link between coffee and longer life. Hypoth... | neutral | 0.6667 | 0.5794 | False | 0.6 | False |
| sar06 | That was a really clever idea. | not_sarcastic | 0.6667 | 0.9183 | False | 0.6 | False |
| nli09 | Premise: A man in a suit is reading a newspaper on a train. Hypothesis... | neutral | 0.7333 | 0.5279 | False | 0.7 | False |
| qa02 | Which gas do plants absorb from the air for photosynthesis? | dioxide | 0.7333 | 0.2824 | False |  |  |
| qa05 | How many sides does a hexagon have? | 6 | 0.7333 | 0.3165 | False |  |  |

## Files
item_metrics.csv, flagged_cases.csv, raw_responses.csv, summary.json, agreement_hist.png, llm_vs_human_entropy.png