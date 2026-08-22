# ETL – Job Fraud Detection

An end-to-end ETL and machine learning pipeline that detects likely-fraudulent
job postings. Raw job listing data is cleaned and transformed into structured
fraud-signal features, loaded into a relational database, and used to train
a classifier that flags suspicious postings.

## Data source

[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
— Kaggle dataset, ~17,880 labeled job postings (`fraudulent` column: 0 = real, 1 = fraudulent).
Not scraped — this project starts from a pre-collected, pre-labeled CSV, so it
demonstrates the **Transform + Load** stages of ETL plus a classification
model on top, not web extraction.

## Tech stack

- **Language:** Python
- **ETL:** pandas, regex-based feature engineering
- **Database:** SQLite (schema portable to PostgreSQL/MySQL)
- **Modeling:** scikit-learn — Logistic Regression, class-balanced
- **Environment:** uv (dependency + virtual environment management)

## Architecture
fake_job_postings.csv (Kaggle, 17,880 rows)
|
v
Transform (etl/transform.py)

engineer 8 rule-based fraud-signal features from text
|
v
data/processed/job_postings_clean.csv
|
v
Load (etl/load.py) --> SQLite database (sql/schema.sql)
|
v
Train (model/train_model.py) --> Logistic Regression classifier


## Feature engineering

Raw text fields (`description`, `requirements`, `benefits`, `company_profile`,
`title`) are scanned for patterns strongly associated with fraudulent listings:

| Feature | Signal |
|---|---|
| `flag_no_salary` | Salary range omitted |
| `flag_no_company_logo` | No company branding on the post |
| `flag_no_company_profile` | Empty "about us" section |
| `flag_urgency_language` | "urgent", "apply immediately", "act now" |
| `flag_requests_personal_info` | SSN, bank account, wire transfer mentions |
| `flag_off_platform_contact` | Personal email/phone/WhatsApp instead of platform messaging |
| `flag_vague_title_language` | "earn $X/week", "no experience needed" |
| `flag_short_description` | Under 30 words — too thin to be a real posting |

These flags are summed into a `fraud_signal_count` per posting and used as
model inputs alongside `description_word_count`.

## Evaluation results

Trained on an 80/20 stratified split (14,304 train / 3,576 test rows).
          precision    recall  f1-score   support

       0       0.98      0.84      0.90      3403
       1       0.16      0.61      0.25       173

accuracy                           0.83      3576

macro avg 0.57 0.72 0.58 3576
weighted avg 0.94 0.83 0.87 3576


## Confusion matrix
[[2847 556]
[ 67 106]]


### Feature importance (logistic regression coefficients)
flag_vague_title_language +3.298
flag_no_company_profile +1.752
flag_off_platform_contact -1.411
flag_no_company_logo +0.833
flag_no_salary -0.793
flag_short_description +0.413
flag_requests_personal_info +0.385
flag_urgency_language -0.182
description_word_count +0.001


### Interpretation

The model catches **61% of actual fraudulent postings** (recall) using just
8 simple rule-based flags — a solid result for a first-pass feature set with
no NLP beyond keyword matching. Precision on the fraud class is low (16%),
meaning it also flags many legitimate postings — largely because signals
like "no salary listed" or "no company logo" are common in real junior/startup
job posts too, not just fraud. This precision/recall tradeoff is expected
given the simplicity of the features, and is a natural next step to improve
(e.g. TF-IDF or embeddings on the raw description text, rather than fixed
keyword rules).
Two counterintuitive findings worth flagging: `flag_off_platform_contact`
and `flag_no_salary` both came out with **negative** coefficients, meaning
they slightly *reduce* predicted fraud probability in this dataset — the
opposite of what I expected going in. This is likely a quirk of this
specific dataset (e.g. legitimate remote-first companies also skipping
salary ranges) and would be worth investigating with a larger or more
recent dataset.

## How to run

```bash
uv venv
.venv\Scripts\Activate.ps1     # Windows PowerShell
uv pip install -r requirements.txt

# place fake_job_postings.csv in data/raw/, then:
cd etl
python transform.py ../data/raw/fake_job_postings.csv
python load.py ../data/processed/job_postings_clean.csv
cd ../model
python train_model.py ../data/processed/job_postings_clean.csv
```

## Project structure
etl/transform.py Cleans raw CSV, engineers fraud-signal features
etl/load.py Loads clean data into SQLite
sql/schema.sql Normalized schema + example SQL queries
model/train_model.py Trains and evaluates the fraud classifier
requirements.txt Python dependencies