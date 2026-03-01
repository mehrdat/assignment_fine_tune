# Financial Sentiment Analysis: BERT vs FinBERT — Before & After Fine-Tuning

## Objective

This project investigates how well transformer-based language models can classify the sentiment of financial news headlines into **negative**, **neutral**, or **positive** categories.

We compare two models:

| Model | Description |
|---|---|
| **`bert-base-uncased`** | A general-purpose BERT model pre-trained on Wikipedia and BookCorpus. It has no prior exposure to financial language. |
| **`ProsusAI/finbert`** | A BERT model further pre-trained on a large financial corpus (financial news, reports). It is domain-adapted for financial NLP tasks. |

Each model is evaluated in two stages:

1. **Before fine-tuning** — used directly as a zero-shot / off-the-shelf classifier.
2. **After fine-tuning** — trained on our labelled financial dataset using the HuggingFace `Trainer` API.

The goal is to answer:
- Does domain-specific pre-training (FinBERT) outperform general-purpose BERT on financial text?
- How much does fine-tuning improve each model?
- Which model + training strategy yields the best overall performance?

---

## Dataset

- **Source**: `resources/cn7050data.csv` — financial news headlines with human-annotated sentiment labels.
- **Columns**: `sentiment` (negative / neutral / positive), `text` (headline string).
- **Size**: ~4,846 rows after removing duplicates.
- **Class distribution** (imbalanced):

| Class | Count | Proportion |
|---|---|---|
| Neutral | 2,879 | ~59.4 % |
| Positive | 1,363 | ~28.1 % |
| Negative | 604 | ~12.5 % |

---

## Methodology

### 1. Data Preparation

```
Load CSV → Remove duplicates → Lowercase labels → Stratified split (80/10/10)
```

- **Training set**: 80 % — used to fine-tune the models.
- **Validation set**: 10 % — used for early stopping and epoch-level evaluation during training.
- **Test set**: 10 % — held out entirely; used only for final evaluation.

Stratified splitting ensures every split mirrors the original class distribution.

### 2. Baseline Evaluation (Before Fine-Tuning)

Both models are loaded via the HuggingFace `pipeline("text-classification")` and run directly on the test set **without any training**.

- **FinBERT** outputs `negative` / `neutral` / `positive` natively (aligned with our labels).
- **BERT-base-uncased** outputs generic `LABEL_0` / `LABEL_1` labels (only 2 classes by default), which are mapped to our 3-class scheme for fair comparison.

### 3. Tokenisation

Each model uses its own tokeniser. The text is tokenised with:
- `truncation=True`
- `padding="max_length"`
- `max_length=128`

The HuggingFace `datasets` library is used to map the tokenisation across train, validation, and test splits for each model independently.

### 4. Fine-Tuning

Both models are fine-tuned using the HuggingFace `Trainer` with identical hyperparameters to ensure a fair comparison:

| Hyperparameter | Value |
|---|---|
| Learning rate | 2 × 10⁻⁵ |
| Batch size (train & eval) | 16 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Optimiser | AdamW (default) |
| Best model selection | `load_best_model_at_end=True` |
| Evaluation strategy | Per epoch |

**Device priority**: CUDA → MPS (Apple Silicon) → CPU.

After training, each model is saved locally:
- `./fine_tuned_bert-base-uncased/`
- `./fine_tuned_ProsusAI_finbert/`

### 5. Evaluation & Comparison

All four scenarios — (FinBERT, BERT-base) × (Before FT, After FT) — are evaluated on the **same test set** using:

#### Global Metrics
| Metric | What it Measures |
|---|---|
| Accuracy | Overall correct predictions / total |
| Balanced Accuracy | Average per-class recall — useful for imbalanced data |
| Weighted F1 | F1 weighted by class support |
| Macro F1 | Unweighted average F1 across classes — treats all classes equally |
| Micro F1 | Aggregated TP/FP/FN across classes (= Accuracy for single-label) |
| Weighted Precision | Precision weighted by class support |
| Macro Precision | Unweighted average precision |
| Weighted Recall | Recall weighted by class support |
| Macro Recall | Unweighted average recall |
| MCC (Matthews Correlation Coefficient) | Balanced measure even with class imbalance; ranges from −1 to +1 |
| Cohen's Kappa | Agreement beyond chance between predicted and true labels |

#### Per-Class Metrics
- Precision, Recall, and F1 for each of `negative`, `neutral`, `positive`.

---

## Visualisations

The notebook generates the following plots, designed to be paper-ready:

| # | Plot | Purpose |
|---|---|---|
| 1 | **Confusion Matrices (2×2 grid)** | Raw TP/FP/FN counts for all 4 scenarios side by side |
| 2 | **Grouped Bar Chart — Key Metrics** | Accuracy, Balanced Accuracy, Weighted F1, Macro F1, MCC, Cohen's Kappa with annotated values |
| 3 | **Per-Class F1 Bar Chart** | Compares F1 per class across all 4 scenarios |
| 4 | **Precision & Recall Heatmap** | Per-class precision and recall in a colour-coded matrix |
| 5 | **Improvement Delta Chart** | Shows the metric change (After − Before) for each model |
| 6 | **Training Loss Curves** | Training/validation loss, validation accuracy, and macro F1 across epochs |
| 7 | **Normalised Confusion Matrices** | Confusion matrices with both count and percentage annotations |
| 8 | **Radar / Spider Charts** | Metric profile overlay (before vs after) for each model |
| 9 | **Class Distribution Chart** | Actual test distribution vs predicted distribution (before & after FT) |

A final **summary table** is rendered with colour-coded best/worst values and exported as:
- `model_comparison_results.csv`
- LaTeX table (printed, ready to copy into a paper)

---

## Project Structure

```
assignment/
├── fine_bert.ipynb              # Main notebook (all code, training, evaluation, plots)
├── requirements.txt             # Python dependencies
├── resources/
│   └── cn7050data.csv           # Dataset
├── fine_tuned_bert-base-uncased/   # Saved fine-tuned BERT model (generated)
├── fine_tuned_ProsusAI_finbert/    # Saved fine-tuned FinBERT model (generated)
├── model_comparison_results.csv    # Exported metrics table (generated)
├── LICENSE
└── README.md
```

---

## Setup & Reproducibility

### Requirements

- Python 3.12+
- macOS (MPS), Linux (CUDA), or CPU

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd assignment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running

Open `fine_bert.ipynb` in Jupyter / VS Code and **run all cells from top to bottom**. The notebook is sequential — each cell depends on variables from earlier cells.

> **Note**: Fine-tuning takes approximately 10–30 minutes depending on hardware (GPU ≪ MPS < CPU).

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.2.2 | PyTorch backend |
| `transformers` | 4.44.0 | Model loading, Trainer API |
| `accelerate` | 0.33.0 | Training acceleration |
| `datasets` | 4.6.1 | HuggingFace dataset handling |
| `scikit-learn` | 1.8.0 | Metrics (F1, MCC, Kappa, etc.) |
| `pandas` | 3.0.1 | Data manipulation |
| `matplotlib` | 3.10.8 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |

---

## Interpretation & Discussion

> _This section is intentionally left for you to fill in with your own analysis based on the results._

### Guiding Questions

1. **Baseline comparison**: How did FinBERT perform out-of-the-box compared to BERT-base-uncased? Why might there be such a large gap before any fine-tuning?

2. **Effect of fine-tuning**: Which model benefited more from fine-tuning — the domain-specific FinBERT or the general-purpose BERT? What does the improvement delta chart show?

3. **Per-class performance**: Which sentiment class was hardest to predict? Does this correlate with class imbalance (negative has the fewest samples)? How did fine-tuning improve minority class recall?

4. **Domain pre-training vs fine-tuning**: Is domain-specific pre-training (FinBERT) more valuable than fine-tuning a general model on task-specific data? Or does fine-tuning close the gap?

5. **MCC and Cohen's Kappa**: These metrics account for class imbalance. Do they tell a different story than accuracy alone?

6. **Training dynamics**: Did either model show signs of overfitting (training loss dropping while validation loss rises)? Were 3 epochs sufficient?

7. **Practical implications**: For a real-world financial NLP pipeline, which model would you deploy and why? Consider accuracy, training cost, and inference speed.

8. **Limitations**: What are the limitations of this study? (e.g., dataset size, single random seed, limited hyperparameter search, no cross-validation)

---

## License

See [LICENSE](LICENSE) for details.
