
import csv
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_FILE = PROJECT_ROOT / "data" / "skills.csv"


def extract_pdf_text(pdf_file):
    """Extract readable text from a PDF resume."""
    try:
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)

        reader = PdfReader(pdf_file)

        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected.")

        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        extracted_text = "\n".join(pages_text).strip()

        if not extracted_text:
            raise ValueError(
                "No readable text found. The PDF may be scanned "
                "or contain images instead of selectable text."
            )

        return extracted_text

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read this PDF: {error}"
        ) from error


def extract_contact_info(text):
    """Extract basic email and phone information."""

    email_matches = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text,
    )

    phone_matches = re.findall(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
        text,
    )

    return {
        "email": email_matches[0] if email_matches else "Not found",
        "phone": phone_matches[0].strip() if phone_matches else "Not found",
    }


def load_skills():
    """Load skills from the local CSV database."""

    if not SKILLS_FILE.exists():
        return []

    with SKILLS_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "skill" not in reader.fieldnames:
            raise ValueError(
                "skills.csv must contain a column named 'skill'."
            )

        return [
            row["skill"].strip()
            for row in reader
            if row.get("skill") and row["skill"].strip()
        ]


def extract_candidate_info(text):
    """Extract contact information and skills from resume text."""

    contact = extract_contact_info(text)
    available_skills = load_skills()
    found_skills = []

    for skill in available_skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, flags=re.IGNORECASE):
            found_skills.append(skill)

    return {
        "email": contact["email"],
        "phone": contact["phone"],
        "skills": found_skills,
    }
