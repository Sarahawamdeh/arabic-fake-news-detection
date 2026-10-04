# Arabic Fake News Detection (AraBERT)

Binary classification of Arabic news articles as **credible** or **not credible**, comparing a classic TF-IDF + Logistic Regression baseline with a fine-tuned **AraBERT** transformer.

Team project for course CIS492 at Yarmouk University (team of three). Team lead: Sarah, who implemented the pipeline and wrote the project documentation.

## Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 75.05% | 74.89% |
| AraBERT (fine-tuned) | **85.91%** | **85.90%** |

Evaluated on a 10,000-article held-out set.

## Dataset

[AFND - Arabic Fake News Dataset](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd) (Kaggle): 606,912 articles scraped from 134 Arab news sources.

- Dropped the `undecided` label, leaving 374,543 articles (207,310 credible / 167,233 not credible).
- Sampled 50,000 articles (`random_state=42`) and split 40,000 train / 10,000 test.

## Method

1. **Cleaning:** removed URLs, punctuation, digits and extra whitespace from the Arabic text.
2. **Baseline:** TF-IDF features + Logistic Regression (scikit-learn).
3. **AraBERT:** fine-tuned `aubmindlab/bert-base-arabertv02` with Hugging Face `Trainer` - 2 epochs, batch size 16, max sequence length 128, weight decay 0.01, 500 warmup steps.
4. **Evaluation:** accuracy, weighted F1, per-class report and confusion matrix.
5. **Inference demo:** a `check_news()` helper returns the predicted label with a confidence score.

## Run it

The notebook is designed for **Google Colab with a GPU runtime**.

1. Create a Kaggle API token and add two Colab Secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY` (key icon in the left sidebar).
2. Open `notebooks/arabic_fake_news_detection.ipynb` in Colab and run the cells in order.

To run locally, install the dependencies with `pip install -r requirements.txt` and provide Kaggle credentials as environment variables.

## Repository structure

```
notebooks/   Colab notebook (data prep, baseline, fine-tuning, evaluation, demo)
docs/        Project documentation (docx) and presentation (pptx)
```

## Limitations

- AFND labels are assigned **per news source**, so the model may partly learn source style rather than the truthfulness of individual claims. A source-level train/test split would be a stricter test.
- Trained on a 50,000-article sample for 2 epochs with texts truncated to 128 tokens; more data, longer training and longer inputs would likely improve results.
- The held-out set was also used for epoch selection (`load_best_model_at_end`); a separate validation split would give a cleaner final estimate.
