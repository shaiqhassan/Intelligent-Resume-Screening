
from pathlib import Path

from app.resume_parser import (
    extract_pdf_text,
    extract_candidate_info,
)


PDF_PATH = Path(__file__).resolve().parent / "sample_resume.pdf"


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Sample resume not found: {PDF_PATH}"
        )

    with PDF_PATH.open("rb") as pdf_file:
        text = extract_pdf_text(pdf_file)

    candidate_info = extract_candidate_info(text)

    print("\n=== EXTRACTED RESUME TEXT ===")
    print(text)

    print("\n=== CANDIDATE INFORMATION ===")
    print(f"Email: {candidate_info['email']}")
    print(f"Phone: {candidate_info['phone']}")
    print(f"Skills: {', '.join(candidate_info['skills']) or 'None found'}")


if __name__ == "__main__":
    main()
