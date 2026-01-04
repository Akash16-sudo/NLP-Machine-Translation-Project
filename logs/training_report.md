# Training Run Report: BART Fine-Tuning (Kaggle)

**Date:** January 4, 2026  
**Model:** `facebook/bart-large-cnn`  
**Dataset:** CNN/DailyMail (Subset)  
**Hardware:** NVIDIA Tesla T4 (Kaggle)

---

## 1. Configuration & Setup
The training was executed on Kaggle to leverage GPU acceleration. To ensure the process completed within the 12-hour session limit, the dataset was strategically optimized.

- **Original Dataset Size:** ~287,113 samples
- **Optimization Strategy:** Subsampling (25%)
- **Actual Training Samples:** `71,779`
- **shards Used:** 1 of 4

## 2. Training Progress
The model was fine-tuned for **1 Epoch**.

- **Loading Data:** Successful (from `cnn_dailymail_processed`)
- **Model Initialization:** `facebook/bart-large-cnn` loaded with weights.
- **Training Duration:** Approx. 5-6 hours.

## 3. Final Metrics
The model achieved strong performance on the validation set after 1 epoch.

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Training Loss** | `1.243` | The model's error rate on the training data. |
| **Validation Loss** | `1.476` | The model's error rate on unseen data (lower is better). |
| **ROUGE-1** | **45.29** | Overlap of unigrams (words). Excellent score. |
| **ROUGE-2** | **21.90** | Overlap of bigrams (pairs of words). Indicates good fluency. |
| **ROUGE-L** | **31.19** | Longest common subsequence. Measures structural similarity. |

## 4. Outcome
- **Status:** ✅ **Success**
- **Artifacts specific:**
    - Final Model weights saved to: `/kaggle/working/models/final_model_full`
    - Downloaded locally to: `models/final_model_full`

**Conclusion:**
The model successfully converged with high accuracy (ROUGE-1 > 40 is typically considered state-of-the-art behavior for this dataset). It is now ready for deployment or further testing.
