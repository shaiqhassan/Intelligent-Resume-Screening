# Intelligent Resume Screening Using NLP and Machine Learning

An AI-powered resume screening application that analyzes PDF resumes, compares them with a job description, identifies known skills, and ranks candidates using semantic similarity.

**Developed by Shaiq Hassan**

## Features

- **PDF Resume Parsing:** Extracts readable text from multiple PDF resumes.
- **Semantic Matching:** Uses Sentence Transformers to compare resumes with job descriptions.
- **Candidate Ranking:** Ranks candidates by cosine similarity.
- **Skill Detection:** Identifies skills using a customizable local CSV database.
- **Contact Extraction:** Attempts to extract email addresses and phone numbers.
- **Resume Quality Checks:** Warns about missing contact details and undetected skills.
- **Visual Dashboard:** Displays similarity scores, rankings, and candidate statistics.
- **Resume Text Preview:** Allows users to inspect extracted text.
- **Skill Comparison:** Shows job-description skills detected or not detected in each resume.
- **CSV Reports:** Exports candidate results, summary statistics, and skill-frequency information.
- **Filtered Export:** Downloads only the candidates currently displayed after filtering.
- **Local Model Support:** Can perform semantic matching offline when the model and required dependencies are available locally.

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
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Intelligent-Resume-Screening
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app/main.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Offline Model Usage

The application uses Sentence Transformers for semantic matching.

The model must be downloaded or cached locally before offline use. A new computer may need an initial model download or a separately supplied local model directory. The Google Font used by the interface is optional and may require internet access.

## Testing

Run the resume parser test:

```bash
python -m app.test_resume_parser
```

Run the semantic matching test:

```bash
python -m app.test_matcher
```

## Important Limitations

- Similarity scores measure textual relevance; they are not probabilities of job success.
- Skill detection relies on a local keyword list and may miss synonyms or differently phrased skills.
- Scanned PDFs containing only images may require OCR.
- Contact extraction is heuristic and may not identify every email address or phone number.
- Candidate rankings should support human review, not replace it.

## Author

**Shaiq Hassan**

Project: Intelligent Resume Screening Using Natural Language Processing and Machine Learning