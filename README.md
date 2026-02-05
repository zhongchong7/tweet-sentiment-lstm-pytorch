# Tweet Sentiment Analysis (PyTorch LSTM)

This project trains a **sentiment classifier** on tweets using a **PyTorch Embedding + LSTM** model inside a Jupyter notebook.

## What’s inside

- **Notebook**: `sentiment_analysis.ipynb`
  - NLTK tokenization + stopword removal
  - Vocabulary building (train split only)
  - Variable-length batching with padding + packed sequences
  - Class-weighted loss to reduce majority-class collapse
  - Evaluation with accuracy + `classification_report`
  - Optional word cloud / fallback “top words” plots

## Dataset

The notebook expects a CSV file named:

- `Tweets.csv`

with at least these columns:
- `text`
- `sentiment` (e.g., `negative`, `neutral`, `positive`)

If you plan to publish this repo, consider adding `Tweets.csv` to `.gitignore` (datasets are often large and/or not redistributable).

## Requirements

- Python 3.9+ (3.10/3.11 also fine)
- Core packages:
  - `torch`
  - `pandas`
  - `numpy`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`
- Optional:
  - `wordcloud` (only needed for true wordcloud plots)

Install dependencies:

```bash
pip install torch pandas numpy nltk scikit-learn matplotlib
```

Optional word clouds:

```bash
pip install wordcloud
```

## How to run

1. Put `Tweets.csv` in the same folder as the notebook.
2. Open `sentiment_analysis.ipynb` in Jupyter/VS Code/Cursor.
3. Run cells top-to-bottom.

Notes:
- The notebook downloads required NLTK resources (`punkt`, `punkt_tab`, `stopwords`) into `~/nltk_data` if needed.
- Training speed depends on whether you have CUDA available.

## Output

After training, the notebook prints:
- **Epoch logs** (train/test loss + accuracy)
- **Accuracy score**
- **Classification report** (precision/recall/F1 per class)

## License

Add a license if you plan to share publicly (e.g., MIT).

