
from app.matcher import rank_resumes


def main():
    job_description = """
    We are looking for a Python developer with experience in
    machine learning, data analysis, NLP, and building AI applications.
    Knowledge of pandas, scikit-learn, and Python programming is preferred.
    """

    resumes = [
        {
            "candidate_name": "Candidate A",
            "text": """
            Python developer with experience in machine learning,
            NLP, pandas, scikit-learn, and data analysis.
            Developed AI applications and predictive models.
            """
        },
        {
            "candidate_name": "Candidate B",
            "text": """
            Graphic designer experienced in visual design,
            branding, typography, image editing, and illustration.
            """
        },
        {
            "candidate_name": "Candidate C",
            "text": """
            Data analyst experienced in Python, data analysis,
            pandas, statistics, and machine learning projects.
            """
        },
    ]

    print("Loading the model and comparing resumes...")
    print("The first run may take several minutes to download the model.")

    results = rank_resumes(job_description, resumes)

    print("\n=== CANDIDATE RANKING ===")

    for candidate in results:
        print(
            f"Rank {candidate['rank']}: "
            f"{candidate['candidate_name']} | "
            f"Similarity: {candidate['similarity_score']:.4f} | "
            f"Display score: {candidate['similarity_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
