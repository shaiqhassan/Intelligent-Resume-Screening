
import io
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import streamlit as st

from app.resume_parser import (
    extract_pdf_text,
    extract_candidate_info,
    load_skills,
)
from app.matcher import rank_resumes


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Intelligent Resume Screening | Shaiq Hassan",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_JOB_DESCRIPTION = """
Job Title: Python and Machine Learning Developer

We are looking for a Python Developer with knowledge of
Machine Learning, Natural Language Processing (NLP), and data analysis.

Required skills:
- Python programming
- Machine Learning
- Natural Language Processing
- Pandas and NumPy
- Scikit-learn
- Data cleaning and preprocessing
- Model development and evaluation
- Problem-solving and analytical skills

Responsibilities:
- Develop and test machine learning models.
- Process and analyze datasets using Python.
- Build NLP-based applications.
- Evaluate model performance and document results.

Preferred qualification:
Bachelor's degree in Computer Science, Artificial Intelligence,
or a related field.
"""


# ==================================================
# SESSION STATE
# ==================================================

if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""

if "resume_results" not in st.session_state:
    st.session_state["resume_results"] = None

if "job_skills" not in st.session_state:
    st.session_state["job_skills"] = []

if "quality_summary" not in st.session_state:
    st.session_state["quality_summary"] = {}

if "unreadable_files" not in st.session_state:
    st.session_state["unreadable_files"] = []


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def find_skills(text, available_skills):
    """Find known skills mentioned in text."""

    found = []

    for skill in available_skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)

    return found


def get_match_strength(score):
    """Describe textual similarity, not hiring suitability."""

    if score >= 70:
        return "Higher similarity", "🟢"

    if score >= 45:
        return "Moderate similarity", "🟡"

    return "Lower similarity", "⚪"


def build_report_csv(results_df, job_description, job_skills):
    """
    Build a CSV containing report metadata, score statistics,
    candidate results, and a detected-skill frequency summary.
    """

    output = io.StringIO()

    if results_df.empty:
        summary_rows = [
            ["Metric", "Value"],
            ["Report generated", datetime.now().isoformat(timespec="seconds")],
            ["Candidates analyzed", 0],
            ["Average similarity (%)", ""],
            ["Highest similarity (%)", ""],
            ["Lowest similarity (%)", ""],
            ["Skills identified in job description", ", ".join(job_skills)],
        ]

        pd.DataFrame(
            summary_rows[1:],
            columns=summary_rows[0],
        ).to_csv(output, index=False)

        return output.getvalue().encode("utf-8-sig")

    scores = results_df["similarity_percent"].astype(float)

    summary_rows = [
        ["Report generated", datetime.now().isoformat(timespec="seconds")],
        ["Candidates analyzed", len(results_df)],
        ["Average similarity (%)", round(scores.mean(), 2)],
        ["Highest similarity (%)", round(scores.max(), 2)],
        ["Lowest similarity (%)", round(scores.min(), 2)],
        ["Job-description skills identified", len(job_skills)],
        ["Job-description skill list", ", ".join(job_skills)],
        [
            "Candidates with quality warnings",
            int(results_df["quality_warning_count"].gt(0).sum()),
        ],
    ]

    output.write("REPORT SUMMARY\n")
    pd.DataFrame(
        summary_rows,
        columns=["Metric", "Value"],
    ).to_csv(output, index=False)

    output.write("\nCANDIDATE RESULTS\n")

    candidate_columns = [
        "rank",
        "candidate_name",
        "similarity_score",
        "similarity_percent",
        "email",
        "phone",
        "skills",
        "matched_skills",
        "not_detected_skills",
        "quality_warnings",
        "filename",
    ]

    candidate_export = results_df[candidate_columns].copy()

    candidate_export["matched_skills"] = candidate_export[
        "matched_skills"
    ].apply(lambda items: ", ".join(items))

    candidate_export["not_detected_skills"] = candidate_export[
        "not_detected_skills"
    ].apply(lambda items: ", ".join(items))

    candidate_export["quality_warnings"] = candidate_export[
        "quality_warnings"
    ].apply(lambda items: "; ".join(items))

    candidate_export.to_csv(output, index=False)

    output.write("\nDETECTED SKILL FREQUENCY\n")

    frequency = Counter()

    for skills in results_df["detected_skill_list"]:
        for skill in skills:
            frequency[skill] += 1

    frequency_rows = [
        {
            "Skill": skill,
            "Candidates with skill detected": count,
            "Percentage of candidates (%)": round(
                count / len(results_df) * 100, 2
            ),
        }
        for skill, count in frequency.most_common()
    ]

    if frequency_rows:
        pd.DataFrame(frequency_rows).to_csv(output, index=False)
    else:
        pd.DataFrame(
            columns=[
                "Skill",
                "Candidates with skill detected",
                "Percentage of candidates (%)",
            ]
        ).to_csv(output, index=False)

    return output.getvalue().encode("utf-8-sig")


def build_candidate_csv(results_df):
    """Export only the candidate rows passed to this function."""

    export_df = results_df.drop(
        columns=[
            "resume_text",
            "detected_skill_list",
            "matched_skills",
            "not_detected_skills",
        ],
        errors="ignore",
    ).copy()

    for column in ["quality_warnings"]:
        if column in export_df.columns:
            export_df[column] = export_df[column].apply(
                lambda items: "; ".join(items)
                if isinstance(items, list)
                else str(items)
            )

    return export_df.to_csv(index=False).encode("utf-8-sig")


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        margin-bottom: 25px;
        background: linear-gradient(
            120deg, #111827 0%, #172554 60%, #164e63 100%
        );
        border: 1px solid rgba(147, 197, 253, 0.25);
    }

    .hero-eyebrow {
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(27px, 4vw, 42px);
        line-height: 1.2;
        color: #f8fafc;
        margin: 0 0 14px 0;
        font-weight: 800;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.8;
        max-width: 800px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 750;
        margin-top: 16px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 20px;
        font-size: 14px;
    }

    .developer-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin-top: 10px;
    }

    .developer-label {
        color: #93c5fd;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .developer-name {
        font-size: 19px;
        font-weight: 750;
        margin: 7px 0;
    }

    .developer-description {
        font-size: 12px;
        line-height: 1.7;
        color: #cbd5e1;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        padding: 17px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 15px;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .footer {
        text-align: center;
        padding: 25px 5px 5px 5px;
        margin-top: 35px;
        color: #94a3b8;
        font-size: 12px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 📄 ResumeAI")
    st.caption("Intelligent Resume Screening")
    st.divider()

    st.markdown("### Project overview")
    st.write(
        "Compare PDF resumes with a job description using "
        "semantic similarity, skill detection, and candidate ranking."
    )

    st.markdown("### Technology stack")

    for technology in [
        "Python",
        "Streamlit",
        "Sentence Transformers",
        "Natural Language Processing",
        "scikit-learn",
        "Pandas",
        "PyPDF",
    ]:
        st.markdown(f"- {technology}")

    st.divider()

    st.markdown(
        """
        <div class="developer-card">
            <div class="developer-label">Developed by</div>
            <div class="developer-name">Shaiq Hassan</div>
            <div class="developer-description">
                AI and Machine Learning project<br>
                Built with Python, NLP and Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Local application • No external AI API")


# ==================================================
# HERO
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">AI · NLP · MACHINE LEARNING</div>
        <h1>Intelligent Resume Screening</h1>
        <p>
            Compare resumes, inspect skill coverage, check resume quality,
            and generate useful candidate reports.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# JOB DESCRIPTION
# ==================================================

st.markdown(
    '<div class="section-heading">01 · Define the opportunity</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">'
    'Enter the job requirements that candidates will be compared against.'
    '</div>',
    unsafe_allow_html=True,
)

button_col1, button_col2 = st.columns(2)

with button_col1:
    if st.button(
        "✨ Load sample job description",
        use_container_width=True,
    ):
        st.session_state["job_description"] = SAMPLE_JOB_DESCRIPTION

with button_col2:
    if st.button(
        "Clear job description",
        use_container_width=True,
    ):
        st.session_state["job_description"] = ""

job_description = st.text_area(
    "Job description",
    key="job_description",
    height=220,
    placeholder="Paste the job title, skills, and responsibilities...",
)


# ==================================================
# PDF UPLOAD
# ==================================================

st.markdown(
    '<div class="section-heading">02 · Add candidate resumes</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">'
    'Upload one or more PDF resumes for analysis.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Select PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple files to compare candidates together.",
)

if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} PDF resume(s) selected.")


# ==================================================
# ANALYSIS AND RESUME QUALITY CHECKS
# ==================================================

if st.button(
    "🚀 Analyze and Rank Candidates",
    type="primary",
    use_container_width=True,
):
    if not job_description.strip():
        st.warning("Please enter a job description.")

    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")

    else:
        parsed_resumes = []
        candidate_details = {}
        errors = []
        unreadable_files = []

        progress = st.progress(0, text="Preparing resume analysis...")

        try:
            available_skills = load_skills()
        except Exception as error:
            st.error(f"Could not load the skills database: {error}")
            available_skills = []

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                resume_text = extract_pdf_text(
                    io.BytesIO(uploaded_file.getvalue())
                )

                candidate_info = extract_candidate_info(resume_text)
                candidate_name = Path(uploaded_file.name).stem

                original_name = candidate_name
                suffix = 2

                while candidate_name in candidate_details:
                    candidate_name = f"{original_name}_{suffix}"
                    suffix += 1

                warnings = []

                if candidate_info["email"] == "Not found":
                    warnings.append("Email address not detected")

                if candidate_info["phone"] == "Not found":
                    warnings.append("Phone number not detected")

                if not candidate_info["skills"]:
                    warnings.append("No known skills detected")

                parsed_resumes.append(
                    {
                        "candidate_name": candidate_name,
                        "text": resume_text,
                    }
                )

                candidate_details[candidate_name] = {
                    **candidate_info,
                    "filename": uploaded_file.name,
                    "resume_text": resume_text,
                    "quality_warnings": warnings,
                }

            except Exception as error:
                error_message = f"{uploaded_file.name}: {error}"
                errors.append(error_message)
                unreadable_files.append(uploaded_file.name)

            progress.progress(
                (index + 1) / len(uploaded_files),
                text=f"Checking resumes: {index + 1} of {len(uploaded_files)}",
            )

        progress.empty()

        if errors:
            st.warning(
                f"{len(errors)} file(s) could not be processed. "
                "They are excluded from candidate ranking."
            )

            for error in errors:
                st.write(f"- {error}")

        st.session_state["unreadable_files"] = unreadable_files

        if not parsed_resumes:
            st.session_state["resume_results"] = None
            st.session_state["quality_summary"] = {}
            st.error(
                "No readable resumes were available for matching. "
                "If these are scanned PDFs, OCR may be required."
            )

        else:
            try:
                with st.spinner(
                    "Matching resumes and checking candidate information..."
                ):
                    results = rank_resumes(
                        job_description,
                        parsed_resumes,
                    )

                job_skills = find_skills(
                    job_description,
                    available_skills,
                )

                for result in results:
                    details = candidate_details[result["candidate_name"]]
                    detected_skills = details["skills"]

                    detected_lookup = {
                        skill.casefold()
                        for skill in detected_skills
                    }

                    matched_skills = [
                        skill for skill in job_skills
                        if skill.casefold() in detected_lookup
                    ]

                    not_detected_skills = [
                        skill for skill in job_skills
                        if skill.casefold() not in detected_lookup
                    ]

                    result["email"] = details["email"]
                    result["phone"] = details["phone"]
                    result["skills"] = ", ".join(detected_skills)
                    result["detected_skill_list"] = detected_skills
                    result["matched_skills"] = matched_skills
                    result["not_detected_skills"] = not_detected_skills
                    result["filename"] = details["filename"]
                    result["resume_text"] = details["resume_text"]
                    result["quality_warnings"] = details["quality_warnings"]
                    result["quality_warning_count"] = len(
                        details["quality_warnings"]
                    )

                results_df = pd.DataFrame(results)

                st.session_state["resume_results"] = results_df
                st.session_state["job_skills"] = job_skills
                st.session_state["quality_summary"] = {
                    "uploaded": len(uploaded_files),
                    "processed": len(results_df),
                    "unreadable": len(unreadable_files),
                    "with_warnings": int(
                        results_df["quality_warning_count"].gt(0).sum()
                    ),
                }

                st.success(
                    f"Analysis completed for {len(results_df)} candidate(s)."
                )

            except Exception as error:
                st.error(f"Matching failed: {error}")


# ==================================================
# RESULTS DASHBOARD
# ==================================================

results_df = st.session_state["resume_results"]

if results_df is not None and not results_df.empty:
    st.divider()

    st.markdown(
        '<div class="section-heading">03 · Candidate intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Explore similarity, skill coverage, resume quality, and reports.'
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Candidates analyzed", len(results_df))

    with metric2:
        st.metric(
            "Highest similarity",
            f"{results_df['similarity_percent'].max():.2f}%",
        )

    with metric3:
        st.metric(
            "Average similarity",
            f"{results_df['similarity_percent'].mean():.2f}%",
        )

    st.caption(
        "Similarity measures textual relevance. It is not a probability "
        "of job success or proof of qualification."
    )

    # ----------------------------------------------
    # QUALITY SUMMARY
    # ----------------------------------------------

    st.divider()
    st.markdown("### 🩺 Resume quality overview")

    quality = st.session_state.get("quality_summary", {})

    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Files uploaded", quality.get("uploaded", 0))
    q2.metric("Readable resumes", quality.get("processed", 0))
    q3.metric("Resumes with warnings", quality.get("with_warnings", 0))
    q4.metric("Unreadable files", quality.get("unreadable", 0))

    if quality.get("with_warnings", 0) > 0:
        st.warning(
            "Some readable resumes are missing contact details or "
            "have no skills detected. Review the candidate quality checks."
        )
    else:
        st.success(
            "No missing-contact or empty-skill warnings were detected "
            "in the readable resumes."
        )

    unreadable_files = st.session_state.get("unreadable_files", [])

    if unreadable_files:
        with st.expander("View unreadable or failed files"):
            for filename in unreadable_files:
                st.error(
                    f"{filename}: could not be processed. "
                    "Check that the PDF opens and contains extractable text."
                )

    # ----------------------------------------------
    # SEARCH AND SORT
    # ----------------------------------------------

    search_col, sort_col = st.columns([2, 1])

    with search_col:
        search_term = st.text_input(
            "🔎 Search candidates",
            placeholder="Search name, email, or skill...",
        )

    with sort_col:
        sort_option = st.selectbox(
            "Sort candidates",
            [
                "Highest similarity",
                "Lowest similarity",
                "Candidate name",
            ],
        )

    filtered_df = results_df.copy()

    if search_term.strip():
        searchable_columns = [
            "candidate_name",
            "email",
            "skills",
            "filename",
        ]

        mask = pd.Series(False, index=filtered_df.index)

        for column in searchable_columns:
            mask |= filtered_df[column].astype(str).str.contains(
                search_term,
                case=False,
                na=False,
                regex=False,
            )

        filtered_df = filtered_df[mask]

    if sort_option == "Lowest similarity":
        filtered_df = filtered_df.sort_values(
            "similarity_score",
            ascending=True,
        )

    elif sort_option == "Candidate name":
        filtered_df = filtered_df.sort_values(
            "candidate_name",
            ascending=True,
        )

    else:
        filtered_df = filtered_df.sort_values(
            "similarity_score",
            ascending=False,
        )

    # ----------------------------------------------
    # CANDIDATE TABLE
    # ----------------------------------------------

    st.markdown("### Candidate ranking")

    if filtered_df.empty:
        st.info("No candidates match your search.")

    else:
        st.dataframe(
            filtered_df[
                [
                    "rank",
                    "candidate_name",
                    "similarity_percent",
                    "email",
                    "phone",
                    "skills",
                    "quality_warning_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank"),
                "candidate_name": st.column_config.TextColumn("Candidate"),
                "similarity_percent": st.column_config.NumberColumn(
                    "Similarity (%)",
                    format="%.2f%%",
                ),
                "email": st.column_config.TextColumn("Email"),
                "phone": st.column_config.TextColumn("Phone"),
                "skills": st.column_config.TextColumn(
                    "Detected skills",
                    width="large",
                ),
                "quality_warning_count": st.column_config.NumberColumn(
                    "Quality warnings"
                ),
            },
        )

    # ----------------------------------------------
    # VISUAL MATCH STRENGTH
    # ----------------------------------------------

    st.divider()
    st.markdown("### 📊 Visual match strength")

    for _, candidate in filtered_df.iterrows():
        score = float(candidate["similarity_percent"])
        label, emoji = get_match_strength(score)

        with st.container(border=True):
            top_col, score_col = st.columns([3, 1])

            with top_col:
                st.markdown(
                    f"**{int(candidate['rank'])}. "
                    f"{candidate['candidate_name']}**"
                )
                st.caption(f"{emoji} {label}")

            with score_col:
                st.metric("Similarity", f"{score:.2f}%")

            st.progress(
                max(0.0, min(1.0, score / 100.0)),
                text=f"{score:.2f}% textual similarity",
            )

    # ----------------------------------------------
    # CANDIDATE DEEP DIVE
    # ----------------------------------------------

    st.divider()
    st.markdown("### 🔍 Candidate deep dive")

    job_skills = st.session_state.get("job_skills", [])

    if job_skills:
        st.write("**Skills identified in the job description:**")
        st.write(" · ".join(job_skills))
    else:
        st.info(
            "No skills from the local skills database were found in "
            "the job description."
        )

    for _, candidate in filtered_df.iterrows():
        with st.expander(
            f"#{int(candidate['rank'])} · "
            f"{candidate['candidate_name']} — "
            f"{candidate['similarity_percent']:.2f}% similarity"
        ):
            contact_tab, skills_tab, preview_tab, quality_tab = st.tabs(
                [
                    "Contact details",
                    "Skill comparison",
                    "Resume text preview",
                    "Quality checks",
                ]
            )

            with contact_tab:
                st.write(f"**Email:** {candidate['email']}")
                st.write(f"**Phone:** {candidate['phone']}")
                st.write(f"**PDF:** {candidate['filename']}")

            with skills_tab:
                skill_col1, skill_col2 = st.columns(2)

                with skill_col1:
                    st.markdown("#### ✅ Detected in resume")

                    if candidate["matched_skills"]:
                        for skill in candidate["matched_skills"]:
                            st.success(f"✓ {skill}")
                    else:
                        st.write("No job-description skills detected.")

                with skill_col2:
                    st.markdown("#### 🔎 Not detected")

                    if candidate["not_detected_skills"]:
                        for skill in candidate["not_detected_skills"]:
                            st.warning(f"! {skill}")
                    else:
                        st.success("All listed job skills were detected.")

                st.caption(
                    "Keyword matching can miss synonyms or differently "
                    "phrased skills. Verify the original resume."
                )

                st.metric(
                    "Job-description skills detected",
                    f"{len(candidate['matched_skills'])} / "
                    f"{len(job_skills)}",
                )

            with preview_tab:
                st.caption(
                    "Extracted text preview; original PDF layout is not shown."
                )

                st.text_area(
                    "Extracted resume content",
                    value=str(candidate["resume_text"]),
                    height=300,
                    disabled=True,
                    key=(
                        f"preview_{candidate['rank']}_"
                        f"{candidate['candidate_name']}"
                    ),
                )

            with quality_tab:
                warnings = candidate["quality_warnings"]

                if warnings:
                    st.warning(
                        f"{len(warnings)} quality warning(s) for this resume."
                    )

                    for warning in warnings:
                        st.write(f"- {warning}")
                else:
                    st.success("No configured quality warnings detected.")

                st.caption(
                    "These checks cover text extraction, contact details, "
                    "and known skills. They do not assess the candidate's "
                    "actual qualifications or the truth of resume claims."
                )

    # ----------------------------------------------
    # FEATURE 4: REPORTS AND FILTERED EXPORT
    # ----------------------------------------------

    st.divider()
    st.markdown("### 📑 Export reports")

    st.write(
        "The comprehensive report contains score statistics, candidate "
        "details, quality warnings, and detected-skill frequency."
    )

    report_data = build_report_csv(
        results_df,
        job_description,
        job_skills,
    )

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.download_button(
            "⬇️ Download comprehensive report",
            data=report_data,
            file_name="resume_screening_full_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        filtered_export_data = build_candidate_csv(filtered_df)

        st.download_button(
            f"⬇️ Export filtered candidates ({len(filtered_df)})",
            data=filtered_export_data,
            file_name="resume_screening_filtered_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "The comprehensive report includes all successfully analyzed "
        "candidates. The filtered export includes only the candidates "
        "currently shown by your search and sort controls."
    )


# ==================================================
# ABOUT SECTION
# ==================================================

st.divider()

about_tab, workflow_tab, notes_tab = st.tabs(
    [
        "👨‍💻 About the developer",
        "⚙️ How it works",
        "ℹ️ Important notes",
    ]
)

with about_tab:
    st.markdown("### Developed by Shaiq Hassan")
    st.write(
        "An educational project exploring Natural Language Processing "
        "and Machine Learning for resume analysis and job-description matching."
    )
    st.write(
        "Python · Streamlit · Sentence Transformers · "
        "scikit-learn · Pandas · PyPDF"
    )

with workflow_tab:
    st.markdown("### How the application works")
    st.markdown(
        """
        1. Enter a job description and upload PDF resumes.
        2. Extract text, contact information, and known skills.
        3. Generate semantic embeddings with Sentence Transformers.
        4. Calculate cosine similarity and rank candidates.
        5. Inspect skills, resume quality, and extracted text.
        6. Export the comprehensive report or filtered candidates.
        """
    )

with notes_tab:
    st.markdown("### Responsible use")
    st.write(
        "Similarity scores do not predict job success or prove qualifications."
    )
    st.write(
        "A skill not detected by the keyword matcher may still be present "
        "under different wording."
    )
    st.write(
        "Scanned image-only PDFs may require OCR, which is not implemented."
    )
    st.write(
        "Review resumes directly and use consistent, job-related criteria."
    )
    st.write(
        "The embedding model runs locally when its required files are "
        "available. The optional Google Font may require internet access."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        <strong>Intelligent Resume Screening</strong><br>
        Developed by Shaiq Hassan · Python · NLP · Machine Learning<br>
        Educational project and decision-support demonstration.
    </div>
    """,
    unsafe_allow_html=True,
)
