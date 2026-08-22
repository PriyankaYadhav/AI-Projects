CREATE TABLE IF NOT EXISTS job_postings (
    job_id                      INTEGER PRIMARY KEY,
    title                       TEXT,
    location                    TEXT,
    department                  TEXT,
    employment_type             TEXT,
    required_experience         TEXT,
    required_education          TEXT,
    industry                    TEXT,
    function                    TEXT,
    description_word_count      INTEGER,
    fraud_signal_count          INTEGER,
    flag_no_salary              BOOLEAN,
    flag_no_company_logo        BOOLEAN,
    flag_no_company_profile     BOOLEAN,
    flag_urgency_language       BOOLEAN,
    flag_requests_personal_info BOOLEAN,
    flag_off_platform_contact   BOOLEAN,
    flag_vague_title_language   BOOLEAN,
    flag_short_description      BOOLEAN,
    fraudulent                  BOOLEAN
);

CREATE TABLE IF NOT EXISTS model_predictions (
    job_id          INTEGER PRIMARY KEY REFERENCES job_postings(job_id),
    predicted_prob  REAL,
    predicted_label BOOLEAN,
    model_version   TEXT,
    predicted_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_fraud_signal ON job_postings(fraud_signal_count);
CREATE INDEX IF NOT EXISTS idx_jobs_industry ON job_postings(industry);

-- Example query: which red flags correlate most with confirmed fraud?
--
-- SELECT
--   ROUND(AVG(CASE WHEN flag_no_salary THEN 1.0 ELSE 0 END), 3) AS no_salary_rate,
--   ROUND(AVG(CASE WHEN flag_urgency_language THEN 1.0 ELSE 0 END), 3) AS urgency_rate,
--   ROUND(AVG(CASE WHEN flag_requests_personal_info THEN 1.0 ELSE 0 END), 3) AS personal_info_rate,
--   fraudulent
-- FROM job_postings
-- GROUP BY fraudulent;