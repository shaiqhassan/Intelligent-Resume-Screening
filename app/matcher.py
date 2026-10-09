
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the model locally, without downloading files."""

    # Prefer a model saved inside the project.
    if LOCAL_MODEL_PATH.is_dir():
        return SentenceTransformer(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
        )

    # Otherwise, allow use of an already-cached Hugging Face model.
    return SentenceTransformer(
        MODEL_NAME,
        local_files_only=True,
    )


def calculate_similarity(job_description, resume_texts):
    """Calculate semantic similarity between a job and resumes."""

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    if not resume_texts:
        raise ValueError("Provide at least one resume.")

    if any(
        not isinstance(text, str) or not text.strip()
        for text in resume_texts
    ):
        raise ValueError("Every resume must contain readable text.")

    model = load_embedding_model()

    documents = [job_description.strip()] + [
        text.strip() for text in resume_texts
    ]

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1:],
    )[0]

    return [float(score) for score in scores]


def rank_resumes(job_description, resumes):
    """Rank resumes by semantic similarity."""

    if not resumes:
        raise ValueError("Provide at least one resume.")

    for resume in resumes:
        if not isinstance(resume, dict):
            raise ValueError("Each resume must be a dictionary.")

        if not str(resume.get("candidate_name", "")).strip():
            raise ValueError("Each resume needs a candidate name.")

        if not str(resume.get("text", "")).strip():
            raise ValueError("Every resume must contain readable text.")

    scores = calculate_similarity(
        job_description,
        [resume["text"] for resume in resumes],
    )

    results = []

    for resume, score in zip(resumes, scores):
        results.append({
            "candidate_name": resume["candidate_name"],
            "similarity_score": round(score, 4),
            "similarity_percent": round(
                max(0.0, min(1.0, score)) * 100, 2
            ),
        })

    results.sort(
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results
