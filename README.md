# Intelligent Resume Screening

**Developed by Shaiq Hassan**

An NLP-powered resume screening application that compares PDF resumes against job descriptions and structured role requirements. It provides explainable candidate rankings, skill coverage, score breakdowns, comparison charts, and downloadable reports.

> This application is a decision-support tool. Scores indicate evidence found in resume text; they are not probabilities of job success and must not replace human review.

## Features

- **PDF Resume Parsing:** Extract text from multiple text-based PDF resumes.
- **Semantic Similarity:** Compare job descriptions and resumes using a locally available Sentence Transformer model.
- **Weighted Candidate Scoring:** Combine semantic similarity, required-skill coverage, project relevance, and optional experience and education evidence.
- **Configurable Weights:** Adjust scoring importance; active weights are normalized to 100%.
- **Job-Specific Criteria:** Select required and preferred skills, minimum experience, and minimum education.
- **Skill Matching:** View required skills detected and not detected in each resume.
- **Structured Evidence:** Detect explicitly stated experience duration and broad education levels when possible.
- **Explainable Results:** Inspect individual scores, explanations, and available-weight coverage.
- **Candidate Comparison:** Compare candidates side by side.
- **Analytics Dashboard:** Explore score charts, skill coverage, and resume-quality indicators.
- **Summary Report:** View top-ranked candidates and key screening statistics.
- **Contact Extraction:** Attempt to extract email addresses and phone numbers.
- **Resume Quality Checks:** Flag missing contact details and missing detected skills.
- **Resume Text Preview:** Inspect extracted resume text.
- **CSV Exports:** Download the full candidate report and score breakdown.
- **Local Model Processing:** No external AI API is used for resume matching.

## Technology Stack

- Python
- Streamlit
- Sentence Transformers
- scikit-learn
- Pandas
- NumPy
- PyPDF

## Project Structure

```text
Intelligent Resume screening/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── matcher.py
│   ├── resume_parser.py
│   ├── test_matcher.py
│   └── test_resume_parser.py
├── data/
│   └── skills.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/shaiqhassan/Intelligent-Resume-Screening.git
cd Intelligent-Resume-Screening
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app/main.py
```

Streamlit will display a local URL, commonly `http://localhost:8501`.

## How to Use

1. Enter a job description or load the sample role.
2. Review the detected skills and select the skills that are genuinely required.
3. Optionally set minimum experience and education criteria.
4. Adjust scoring weights in the sidebar.
5. Upload one or more text-based PDF resumes.
6. Select **Analyze candidates**.
7. Review rankings, score breakdowns, matched and not-detected skills, candidate comparisons, analytics, and the summary report.
8. Download the CSV reports if required.

## Scoring Overview

| Component | Default weight |
|---|---:|
| Semantic similarity | 35% |
| Required-skill coverage | 30% |
| Project relevance | 15% |
| Experience match | 10% |
| Education match | 10% |

Weights are configurable and normalized to total 100%. If a component cannot be evaluated, it is excluded for that candidate and the remaining applicable weights are normalized. Check the available-weight indicator when comparing scores.

- **Semantic similarity:** Measures semantic relevance between the job description and resume.
- **Required-skill coverage:** Percentage of selected required skills detected in resume text.
- **Project relevance:** Semantic relevance of identifiable project-related text.
- **Experience match:** Compares explicitly stated experience with the selected minimum when detectable.
- **Education match:** Compares a detected broad education level with the selected minimum when possible.

## Offline Model Usage

The application uses Sentence Transformers for semantic matching and is configured to load the model from local files or cache only. The model must already be available locally. A new computer may require initial model setup before offline inference works.

No external AI API is used by the matching workflow.

## Testing

Activate the project virtual environment and run:

```powershell
python -m pip check
python -m app.test_resume_parser
python -m app.test_matcher
```

## Limitations and Responsible Use

- Scores are text-evidence indicators, not probabilities of job performance.
- Keyword matching may miss synonyms, abbreviations, or differently phrased skills.
- A skill marked **not detected** may still be possessed by the candidate.
- Experience extraction relies on explicit statements and does not reliably calculate duration from complex employment dates.
- Education extraction identifies broad degree levels and does not verify institutions or exact subject equivalence.
- Scanned PDFs may require OCR.
- Contact extraction and quality checks are heuristic.
- Use consistent, job-relevant criteria and review original resumes. Human review remains essential.

## Author

**Shaiq Hassan**

Project: Intelligent Resume Screening Using Natural Language Processing and Machine Learning

GitHub: https://github.com/shaiqhassan/Intelligent-Resume-Screening
