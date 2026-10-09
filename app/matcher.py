"""Explainable, local resume matching with robust skill matching and normalized weights."""
from functools import lru_cache
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"

DEFAULT_WEIGHTS = {
    "semantic_similarity": 35.0,
    "required_skill_coverage": 30.0,
    "project_relevance": 15.0,
    "experience_match": 10.0,
    "education_match": 10.0,
}

def normalize_weights(weights=None):
    """Normalize non-negative weights so configured values always sum to 100."""
    source = dict(weights or DEFAULT_WEIGHTS)
    keys = list(DEFAULT_WEIGHTS)
    cleaned = {key: max(0.0, float(source.get(key, DEFAULT_WEIGHTS[key]))) for key in keys}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one scoring weight must be greater than zero.")
    return {key: value * 100.0 / total for key, value in cleaned.items()}


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load a local model if available; otherwise download it from Hugging Face."""
    if LOCAL_MODEL_PATH.is_dir():
        return SentenceTransformer(str(LOCAL_MODEL_PATH))

    return SentenceTransformer(MODEL_NAME)


def _clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def _normalize_phrase(value):
    """Normalize common punctuation/spacing variations for deterministic phrase matching."""
    value = _clean(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[\u2010-\u2015]", "-", value)
    value = re.sub(r"[^a-z0-9+#.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def _contains_skill(text, skill):
    text_norm = _normalize_phrase(text)
    skill_norm = _normalize_phrase(skill)
    if not skill_norm:
        return False
    # Word-boundary matching avoids matching "R" inside unrelated words,
    # while normalizing punctuation handles variants such as scikit-learn.
    pattern = r"(?<![a-z0-9])" + re.escape(skill_norm) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text_norm))

def _similarity(a, b):
    if not _clean(a) or not _clean(b):
        return None
    model = load_embedding_model()
    vectors = model.encode([a, b], normalize_embeddings=True)
    score = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
    return round(max(0.0, min(100.0, score * 100.0)), 2)

def calculate_similarity(job_description, resume_texts):
    """Compatibility API returning cosine similarity values on a 0–1 scale."""
    if not _clean(job_description):
        raise ValueError("Job description cannot be empty.")
    if not resume_texts:
        raise ValueError("At least one resume text is required.")
    model = load_embedding_model()
    vectors = model.encode([job_description] + list(resume_texts), normalize_embeddings=True)
    base = vectors[0]
    return [float(cosine_similarity([base], [vector])[0][0]) for vector in vectors[1:]]

def extract_experience_years(text):
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s*[:\-]\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    ]
    values = []
    for pattern in patterns:
        values.extend(float(m.group(1)) for m in re.finditer(pattern, (text or "").lower()))
    return max(values) if values else None

def extract_education_level(text):
    checks = [
        ("PhD/Doctorate", r"\b(ph\.?d|doctorate|doctoral)\b"),
        ("Master's", r"\b(master'?s|m\.?sc|m\.?tech|m\.?e\.|mba|mca)\b"),
        ("Bachelor's", r"\b(bachelor'?s|b\.?sc|b\.?tech|b\.?e\.|bca|b\.?eng)\b"),
        ("Diploma", r"\b(diploma|associate degree)\b"),
        ("High school", r"\b(high school|12th pass|higher secondary|senior secondary)\b"),
    ]
    for label, pattern in checks:
        if re.search(pattern, (text or "").lower()):
            return label
    return None

def _education_rank(label):
    return {"High school": 1, "Diploma": 2, "Bachelor's": 3, "Master's": 4, "PhD/Doctorate": 5}.get(label)

def _education_score(level, requirement):
    if not requirement or requirement == "Not specified":
        return None, "Education is not scored because no minimum was specified."
    if not level:
        return None, "Education level could not be reliably detected."
    required = extract_education_level(requirement)
    if not required:
        return None, "Education requirement needs manual review."
    score = 100.0 if _education_rank(level) >= _education_rank(required) else 0.0
    return score, f"Detected {level}; minimum requested: {required}."

def _extract_section(text, headings):
    lines = (text or "").splitlines()
    targets = {x.lower() for x in headings}
    heading = re.compile(r"^\s*(summary|profile|objective|skills|technical skills|experience|work experience|employment|professional experience|education|projects|personal projects|certifications|achievements|responsibilities)\s*:?\s*$", re.I)
    active, output = False, []
    for line in lines:
        label = line.strip().strip(":").lower()
        if not active:
            if label in targets:
                active = True
            continue
        if heading.match(line.strip()):
            break
        output.append(line)
    return _clean(" ".join(output))

def _skill_scores(text, required_skills, preferred_skills):
    matched_required = [s for s in required_skills if _contains_skill(text, s)]
    missing_required = [s for s in required_skills if s not in matched_required]
    matched_preferred = [s for s in preferred_skills if _contains_skill(text, s)]
    missing_preferred = [s for s in preferred_skills if s not in matched_preferred]
    required_score = (100.0 * len(matched_required) / len(required_skills)) if required_skills else None
    return required_score, matched_required, missing_required, matched_preferred, missing_preferred

def rank_resumes(job_description, resumes, required_skills=None, preferred_skills=None,
                 minimum_years=0, education_requirement="Not specified", weights=None):
    if not _clean(job_description):
        raise ValueError("Job description cannot be empty.")
    if not resumes:
        raise ValueError("At least one resume is required.")
    normalized_weights = normalize_weights(weights)
    required_skills = list(dict.fromkeys(_clean(s) for s in (required_skills or []) if _clean(s)))
    preferred_skills = list(dict.fromkeys(_clean(s) for s in (preferred_skills or []) if _clean(s)))
    results = []

    for candidate in resumes:
        text = candidate.get("text", "")
        if not _clean(text):
            raise ValueError("Each resume must contain non-empty candidate text.")
        semantic = _similarity(job_description, text)

        req_score, matched_req, missing_req, matched_pref, missing_pref = _skill_scores(
            text, required_skills, preferred_skills)

        project_text = _extract_section(text, ["projects", "personal projects"])
        if not project_text:
            project_text = _clean(" ".join(line for line in text.splitlines()
                if re.search(r"\b(project|developed|built|implemented)\b", line, re.I)))
        project_score = _similarity(job_description, project_text) if project_text else None

        years = extract_experience_years(text)
        if minimum_years > 0 and years is not None:
            experience_score = min(100.0, 100.0 * years / minimum_years)
            experience_note = f"Detected explicit experience of {years:g} years; requirement is {minimum_years}."
        elif minimum_years > 0:
            experience_score = None
            experience_note = "Experience not detected reliably; this component is excluded from the total."
        else:
            experience_score = None
            experience_note = "No minimum experience was set; this component is excluded from the total."

        education = extract_education_level(text)
        education_score, education_note = _education_score(education, education_requirement)

        components = {
            "semantic_similarity": semantic,
            "required_skill_coverage": req_score,
            "project_relevance": project_score,
            "experience_match": experience_score,
            "education_match": education_score,
        }
        available = {k: v for k, v in components.items() if v is not None and normalized_weights[k] > 0}
        available_weight = sum(normalized_weights[k] for k in available)
        overall = (sum(components[k] * normalized_weights[k] for k in available) / available_weight
                   if available_weight else None)

        explanations = [
            f"Semantic similarity is {semantic:.1f}/100." if semantic is not None else "Semantic similarity unavailable.",
            (f"Required skills detected: {len(matched_req)}/{len(required_skills)}."
             if required_skills else "No required skills selected. Select required skills to enable this component."),
            (f"Matched required skills: {', '.join(matched_req)}." if matched_req else "No selected required skills were detected in the resume."),
            (f"Not detected in resume text: {', '.join(missing_req)}." if missing_req else "All selected required skills were detected." if required_skills else ""),
            ("Project relevance was compared against project-related resume text." if project_text
             else "No clear project section was detected; project relevance is excluded."),
            experience_note, education_note,
        ]
        explanations = [x for x in explanations if x]
        row = dict(candidate)
        row.update(components)
        row.update({
            "overall_match_score": round(overall, 2) if overall is not None else None,
            "available_weight_percent": round(available_weight, 2),
            "normalized_weights": normalized_weights,
            "experience_years_detected": years,
            "education_level_detected": education or "Unknown",
            "matched_required_skills": matched_req,
            "missing_required_skills": missing_req,
            "matched_preferred_skills": matched_pref,
            "missing_preferred_skills": missing_pref,
            "component_scores": components,
            "explanations": explanations,
            "similarity_percent": semantic or 0.0,
            "similarity_score": (semantic or 0.0) / 100.0,
            "matched_skills": matched_req,
            "not_detected_skills": missing_req,
        })
        results.append(row)

    results.sort(key=lambda r: (r["overall_match_score"] is not None,
                                r["overall_match_score"] if r["overall_match_score"] is not None else -1),
                 reverse=True)
    for index, row in enumerate(results, 1):
        row["rank"] = index
    return results
