import re
from collections import Counter

import pandas as pd
import streamlit as st
import sys
from pathlib import Path
# Ensure the project root is available for imports on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.matcher import DEFAULT_WEIGHTS, normalize_weights, rank_resumes
from app.resume_parser import extract_pdf_text, extract_candidate_info, load_skills

st.set_page_config(page_title="ResumeAI | Shaiq Hassan", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")

SAMPLE_JOB = """Job Title: Python and Machine Learning Developer

Required skills: Python, Machine Learning, Natural Language Processing, Pandas, NumPy, Scikit-learn.
Responsibilities: Develop and test machine learning models, analyze datasets, build NLP applications,
evaluate model performance and document results. Preferred qualification: Bachelor's degree in
Computer Science, Artificial Intelligence, or a related field."""

st.markdown("""
<style>
:root{--accent:#7c8cff}
.block-container{max-width:1500px;padding-top:2.8rem;padding-bottom:3.2rem}
header[data-testid="stHeader"]{height:3.25rem;background:rgba(10,14,24,.96)}
div[data-testid="stAppViewBlockContainer"]{padding-top:2.2rem}
.hero{padding:38px 40px 34px;border-radius:22px;margin:8px 0 26px;background:linear-gradient(115deg,#111827 0%,#1e1b4b 52%,#164e63 100%);border:1px solid #334155;box-shadow:0 14px 40px rgba(0,0,0,.18)}
.hero .eyebrow{color:#a5b4fc;text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700;margin-bottom:10px}
.hero h1{color:#f8fafc;font-size:40px;line-height:1.2;margin:0 0 12px;padding:0}
.hero p{color:#dbeafe;margin:0;max-width:900px;line-height:1.75;font-size:16px}
.hero .hero-meta{color:#cbd5e1;margin-top:18px;padding-top:14px;border-top:1px solid rgba(203,213,225,.2);font-size:13px}
.section-kicker{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#94a3b8;font-weight:700}
div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(30,41,59,.75),rgba(15,23,42,.75));border:1px solid rgba(148,163,184,.22);padding:17px;border-radius:16px}
div[data-testid="stMetricLabel"]{color:#cbd5e1}
div[data-testid="stTabs"] button{font-weight:600}
.stButton>button{border-radius:10px;transition:all .15s ease}
.stDownloadButton>button{border-radius:10px}
div[data-testid="stExpander"]{border-radius:12px;border:1px solid rgba(148,163,184,.22)}
.small-note{color:#94a3b8;font-size:13px}
.about-card{padding:18px 20px;border:1px solid rgba(148,163,184,.22);border-radius:14px;background:rgba(30,41,59,.28);margin-bottom:10px}
.about-card h4{margin:0 0 8px;color:#e2e8f0}
.about-card p{margin:0;color:#cbd5e1;line-height:1.65}
</style>
<div class="hero">
 <div class="eyebrow">Local NLP · Explainable scoring · Candidate insights</div>
 <h1>Intelligent Resume Screening</h1>
 <p>An explainable resume evaluation workspace that compares candidate evidence with job requirements,
 breaks down scoring criteria, and helps people make more informed shortlisting decisions.</p>
 <div class="hero-meta">Developed by <strong>Shaiq Hassan</strong> &nbsp;·&nbsp; Local model processing &nbsp;·&nbsp; No external AI API</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✦ ResumeAI")
    st.markdown("**Developed by Shaiq Hassan**")
    st.caption("Local model · No external AI API")
    st.divider()
    st.markdown("### Score weighting")
    st.caption("Set importance. Values are automatically normalized to total 100%.")
    raw_weights = {
        "semantic_similarity": float(st.slider("Semantic relevance", 0, 100, 35, key="w_sem")),
        "required_skill_coverage": float(st.slider("Required skills", 0, 100, 30, key="w_skill")),
        "project_relevance": float(st.slider("Project evidence", 0, 100, 15, key="w_project")),
        "experience_match": float(st.slider("Experience evidence", 0, 100, 10, key="w_exp")),
        "education_match": float(st.slider("Education evidence", 0, 100, 10, key="w_edu")),
    }
    if sum(raw_weights.values()) == 0:
        st.error("Increase at least one weight to continue.")
        effective_weights = None
    else:
        effective_weights = normalize_weights(raw_weights)
        st.markdown("**Effective weights (normalized)**")
        for key, label in [
            ("semantic_similarity", "Semantic"),
            ("required_skill_coverage", "Skills"),
            ("project_relevance", "Projects"),
            ("experience_match", "Experience"),
            ("education_match", "Education"),
        ]:
            st.caption(f"{label}: {effective_weights[key]:.1f}%")
        st.progress(1.0)
    st.divider()
    st.caption("Scores are evidence indicators, not a hiring decision.")

top1, top2 = st.columns([3, 1])
with top1:
    st.markdown('<div class="section-kicker">01 / Role definition</div>', unsafe_allow_html=True)
    st.subheader("Define the job")
# Button callbacks update widget state before Streamlit creates the text area.
def load_sample_role():
    st.session_state["job_description"] = SAMPLE_JOB

def clear_job_role():
    st.session_state["job_description"] = ""

with top2:
    st.write("")
    st.button("↻ Reset example", use_container_width=True,
              on_click=load_sample_role)

job_description = st.text_area("Job description", key="job_description", height=155,
    placeholder="Paste job title, responsibilities and requirements here.")
b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    st.button("Load sample role", use_container_width=True, on_click=load_sample_role)
with b2:
    st.button("Clear role", use_container_width=True, on_click=clear_job_role)
with b3:
    st.markdown('<p class="small-note">For best results, specify the actual requirements rather than relying on the description alone.</p>', unsafe_allow_html=True)

st.markdown("#### Structured criteria")
try:
    skill_catalogue = sorted(set(load_skills()), key=str.lower)
except Exception as exc:
    st.error(f"Could not load data/skills.csv: {exc}")
    skill_catalogue = []

def suggest_skills(description, catalogue):
    """Find catalogue terms explicitly mentioned in the job description."""
    found = []
    normalized = re.sub(r"\s+", " ", (description or "").lower())
    for skill in catalogue:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return found

suggested = suggest_skills(job_description, skill_catalogue)

# Keep the required-skill selection synchronized when the job description changes.
# Do not reset it on every rerun (for example, when the user changes a slider).
previous_job_text = st.session_state.get("_last_job_text_for_skill_detection")
if "required_skills" not in st.session_state:
    st.session_state["required_skills"] = suggested or [
        s for s in ["Python", "Machine Learning", "Natural Language Processing", "Pandas", "NumPy"]
        if s in skill_catalogue
    ]
elif previous_job_text is not None and job_description != previous_job_text:
    st.session_state["required_skills"] = suggested

st.session_state["_last_job_text_for_skill_detection"] = job_description

skill1, skill2 = st.columns(2)
with skill1:
    required_skills = st.multiselect("Required skills", skill_catalogue, key="required_skills",
        help="These selected skills drive the required-skill coverage score.")
    if suggested:
        st.caption("Detected in job description: " + ", ".join(suggested))
        if st.button("Select all detected skills", key="select_detected_skills"):
            st.session_state["required_skills"] = suggested
            st.rerun()
    elif job_description.strip():
        st.caption("No catalogue skills found automatically. Add skills manually from the list.")
with skill2:
    preferred_skills = st.multiselect("Preferred skills (informational)", skill_catalogue,
        help="Reported separately; not included in the weighted score.")

crit1, crit2 = st.columns(2)
with crit1:
    minimum_years = st.number_input("Minimum years of experience (0 = not scored)", min_value=0, max_value=50, value=0, step=1)
with crit2:
    education_requirement = st.selectbox("Minimum education", ["Not specified", "High school", "Diploma", "Bachelor's", "Master's", "PhD/Doctorate"])

st.markdown('<div class="section-kicker">02 / Candidate pool</div>', unsafe_allow_html=True)
st.subheader("Upload resumes")
uploaded_files = st.file_uploader("Add PDF resumes", type=["pdf"], accept_multiple_files=True,
                                  help="Use text-based PDFs. Scanned PDFs may need OCR.")
st.caption(f"{len(uploaded_files) if uploaded_files else 0} resume(s) selected")

analyze = st.button("✦ Analyze candidates", type="primary", use_container_width=True)
if analyze:
    if not job_description.strip():
        st.warning("Enter a job description first.")
    elif not uploaded_files:
        st.warning("Upload at least one PDF resume.")
    elif not required_skills:
        st.warning("Select at least one required skill so the skill score is meaningful.")
    elif effective_weights is None:
        st.warning("Set at least one scoring weight above zero.")
    else:
        candidates, failures = [], []
        progress = st.progress(0, text="Reading candidate resumes…")
        for i, file in enumerate(uploaded_files):
            try:
                text = extract_pdf_text(file)
                if not text.strip():
                    raise ValueError("No extractable text; possibly a scanned PDF.")
                info = extract_candidate_info(text)
                detected = info.get("skills", [])
                warnings = []
                if not info.get("email"): warnings.append("Email not detected")
                if not info.get("phone"): warnings.append("Phone not detected")
                if not detected: warnings.append("No configured skills detected")
                candidates.append({
                    "candidate_name": file.name.rsplit(".", 1)[0],
                    "filename": file.name, "text": text,
                    "email": info.get("email", ""), "phone": info.get("phone", ""),
                    "detected_skill_list": detected, "skills": ", ".join(detected),
                    "quality_warnings": warnings, "quality_warning_count": len(warnings),
                })
            except Exception as exc:
                failures.append({"filename": file.name, "error": str(exc)})
            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")
        progress.empty()
        if candidates:
            try:
                ranked = rank_resumes(job_description, candidates, required_skills, preferred_skills,
                    int(minimum_years), education_requirement, effective_weights)
                st.session_state["resume_results"] = ranked
                st.session_state["analysis_failures"] = failures
                st.session_state["analyzed_role"] = job_description
                st.session_state["analyzed_required_skills"] = required_skills
                st.success(f"Analysis complete — {len(ranked)} candidate(s) evaluated.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
        else:
            st.error("No readable resumes could be analyzed.")
            st.session_state["analysis_failures"] = failures

results = st.session_state.get("resume_results", [])
if results:
    df = pd.DataFrame(results)
    st.divider()
    st.markdown('<div class="section-kicker">03 / Insights</div>', unsafe_allow_html=True)
    st.subheader("Candidate summary")
    scored = [r["overall_match_score"] for r in results if r.get("overall_match_score") is not None]
    semantics = [r["semantic_similarity"] for r in results if r.get("semantic_similarity") is not None]
    metrics = st.columns(4)
    metrics[0].metric("Candidates analyzed", len(results))
    metrics[1].metric("Top weighted score", f"{max(scored):.1f}/100" if scored else "N/A")
    metrics[2].metric("Average weighted score", f"{sum(scored)/len(scored):.1f}/100" if scored else "N/A")
    metrics[3].metric("Avg. semantic similarity", f"{sum(semantics)/len(semantics):.1f}%" if semantics else "N/A")

    t_rank, t_compare, t_graph, t_summary, t_export = st.tabs(
        ["Rankings", "Compare", "Analytics", "Summary report", "Export"])

    score_columns = ["rank", "candidate_name", "overall_match_score", "semantic_similarity",
                     "required_skill_coverage", "project_relevance", "experience_match",
                     "education_match", "available_weight_percent"]
    with t_rank:
        filter_text = st.text_input("Search candidates", placeholder="Name, filename, skills…")
        sort_by = st.selectbox("Sort by", ["overall_match_score", "required_skill_coverage",
            "semantic_similarity", "project_relevance", "experience_match", "education_match"])
        shown = df.copy()
        if filter_text:
            shown = shown[shown.astype(str).apply(
                lambda col: col.str.contains(filter_text, case=False, na=False)).any(axis=1)]
        shown = shown.sort_values(sort_by, ascending=False, na_position="last")
        st.dataframe(shown[[c for c in score_columns if c in shown.columns]],
                     use_container_width=True, hide_index=True)
        st.caption("‘Not detected’ means the text parser did not find evidence; it does not prove a candidate lacks that skill.")
        for row in shown.to_dict("records"):
            with st.expander(f"#{row['rank']} · {row['candidate_name']} · {row['overall_match_score'] if row['overall_match_score'] is not None else 'N/A'}/100"):
                st.write(f"**Email:** {row.get('email') or 'Not detected'}")
                st.write(f"**Phone:** {row.get('phone') or 'Not detected'}")
                st.write("**Required skills detected:** " + (", ".join(row.get("matched_required_skills", [])) or "None detected"))
                st.write("**Required skills not detected:** " + (", ".join(row.get("missing_required_skills", [])) or "None"))
                st.write("**Preferred skills detected:** " + (", ".join(row.get("matched_preferred_skills", [])) or "None detected"))
                for note in row.get("explanations", []):
                    st.markdown("- " + note)
                if row.get("quality_warnings"):
                    st.warning("Resume quality flags: " + "; ".join(row["quality_warnings"]))
                with st.expander("Resume text preview"):
                    st.text_area("Extracted text", row.get("text", ""), height=200,
                                 key="preview_" + row["filename"])

    with t_compare:
        options = {f"#{r['rank']} · {r['candidate_name']}": r["filename"] for r in results}
        selected = st.multiselect("Select up to four candidates", list(options.keys()),
                                  default=list(options.keys())[:2], max_selections=4)
        selected_files = [options[x] for x in selected]
        subset = [r for r in results if r["filename"] in selected_files]
        if subset:
            comp = pd.DataFrame([{
                "Candidate": r["candidate_name"], "Rank": r["rank"],
                "Weighted score": r["overall_match_score"], "Semantic": r["semantic_similarity"],
                "Skills": r["required_skill_coverage"], "Projects": r["project_relevance"],
                "Experience": r["experience_match"], "Education": r["education_match"],
                "Matched skills": ", ".join(r["matched_required_skills"]),
                "Not detected": ", ".join(r["missing_required_skills"]),
            } for r in subset])
            st.dataframe(comp, use_container_width=True, hide_index=True)
            graph = comp.set_index("Candidate")[["Weighted score", "Semantic", "Skills", "Projects", "Experience", "Education"]]
            st.bar_chart(graph)

    with t_graph:
        st.markdown("#### Score components by candidate")
        chart = df[["candidate_name", "overall_match_score", "semantic_similarity",
                    "required_skill_coverage", "project_relevance", "experience_match",
                    "education_match"]].set_index("candidate_name")
        st.bar_chart(chart)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Required skill coverage")
            coverage = Counter(skill for row in results for skill in row.get("matched_required_skills", []))
            if coverage:
                st.bar_chart(pd.DataFrame([{"Skill": k, "Candidates": v} for k, v in coverage.items()]).set_index("Skill"))
            else:
                st.info("No required skills detected yet.")
        with g2:
            st.markdown("#### Resume quality flags")
            st.bar_chart(pd.DataFrame([{"Candidate": r["candidate_name"],
                "Flags": r.get("quality_warning_count", 0)} for r in results]).set_index("Candidate"))

    with t_summary:
        best = results[0]
        avg_score = sum(scored) / len(scored) if scored else 0
        st.markdown("#### Screening summary")
        st.write(f"**Role analyzed:** {st.session_state.get('analyzed_role', '')[:500]}")
        st.write(f"**Candidates processed:** {len(results)}")
        st.write(f"**Highest weighted score:** {best['candidate_name']} ({best['overall_match_score']}/100)")
        st.write(f"**Average weighted score:** {avg_score:.1f}/100")
        st.write(f"**Required criteria:** {', '.join(st.session_state.get('analyzed_required_skills', []))}")
        st.markdown("#### Top candidates")
        st.dataframe(pd.DataFrame([{"Rank": r["rank"], "Candidate": r["candidate_name"],
            "Weighted score": r["overall_match_score"], "Skill coverage": r["required_skill_coverage"],
            "Semantic similarity": r["semantic_similarity"]} for r in results[:5]]),
            use_container_width=True, hide_index=True)
        st.info("This is a decision-support summary. Review the source resumes before making any hiring decision.")

    with t_export:
        export = df.drop(columns=["text", "component_scores", "explanations"], errors="ignore").copy()
        for col in export.columns:
            if export[col].map(lambda x: isinstance(x, list)).any():
                export[col] = export[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
            elif export[col].map(lambda x: isinstance(x, dict)).any():
                export[col] = export[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        st.download_button("Download full screening report (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
            "resume_screening_summary.csv", "text/csv", use_container_width=True)
        breakdown = df[["rank", "candidate_name", "overall_match_score", "semantic_similarity",
            "required_skill_coverage", "project_relevance", "experience_match", "education_match",
            "available_weight_percent"]]
        st.download_button("Download score breakdown (CSV)", breakdown.to_csv(index=False).encode("utf-8-sig"),
            "resume_score_breakdown.csv", "text/csv", use_container_width=True)

failures = st.session_state.get("analysis_failures", [])
if failures:
    with st.expander("Files that could not be analyzed"):
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

st.divider()
st.markdown('<div class="section-kicker">04 / Product guide</div>', unsafe_allow_html=True)
st.subheader("About this application")
about_tabs = st.tabs(["Overview", "Key features", "How to use", "Scoring guide", "Developer & limitations"])

with about_tabs[0]:
    st.markdown("""
    <div class="about-card">
      <h4>Purpose</h4>
      <p><strong>Intelligent Resume Screening</strong> is a locally run, NLP-based decision-support application.
      It compares resume text with a job description and explicit role criteria, then presents a transparent
      score breakdown so a reviewer can inspect the evidence behind each ranking.</p>
    </div>
    """, unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Processing", "Local")
    b.metric("External AI APIs", "None")
    c.metric("Primary output", "Explainable ranking")

with about_tabs[1]:
    feature_cols = st.columns(2)
    with feature_cols[0]:
        st.markdown("""
        <div class="about-card"><h4>Candidate evaluation</h4>
        <p>Semantic relevance, required-skill coverage, project relevance, and optional experience and education evidence.</p></div>
        <div class="about-card"><h4>Transparent matching</h4>
        <p>Matched and not-detected required skills, component scores, and explanations for unavailable evidence.</p></div>
        <div class="about-card"><h4>Candidate comparison</h4>
        <p>Compare multiple candidates side by side with score charts and evidence summaries.</p></div>
        """, unsafe_allow_html=True)
    with feature_cols[1]:
        st.markdown("""
        <div class="about-card"><h4>Analytics and reports</h4>
        <p>Score distribution, skill coverage, resume-quality flags, top-candidate summary, and CSV exports.</p></div>
        <div class="about-card"><h4>Configurable scoring</h4>
        <p>Adjust the importance of each scoring component; active weights are normalized automatically.</p></div>
        <div class="about-card"><h4>Privacy-conscious workflow</h4>
        <p>Model inference runs from a locally available Sentence Transformer model; no external AI API is called by the app.</p></div>
        """, unsafe_allow_html=True)

with about_tabs[2]:
    st.markdown("""
    1. **Define the role:** paste a job description or load the sample role.
    2. **Select required skills:** review the detected suggestions and select the skills that truly are required.
    3. **Set optional criteria:** enter a minimum experience level or education requirement only when relevant.
    4. **Set score weights:** use the sidebar to choose importance. The displayed effective weights are normalized to 100%.
    5. **Upload resumes:** add text-based PDF resumes and select **Analyze candidates**.
    6. **Review results:** inspect rankings, matched/not-detected skills, candidate comparison, analytics, and the summary report.
    7. **Export:** download the full CSV report or the score breakdown for further review.
    """)

with about_tabs[3]:
    st.markdown("""
    The overall score is a weighted combination of the available components:
    - **Semantic relevance:** similarity between the job description and resume text.
    - **Required skills:** share of selected required skills detected in the resume text.
    - **Project evidence:** semantic relevance of text identified as project-related.
    - **Experience evidence:** compares explicitly stated years with the minimum specified, when detectable.
    - **Education evidence:** compares a detected broad degree level with the selected minimum, when detectable.

    If a component cannot be assessed, it is excluded from that candidate's calculation and the remaining active weights are normalized. Therefore, check the **available weight** indicator as well as the final score.
    """)

with about_tabs[4]:
    st.markdown("""
    **Developed by Shaiq Hassan**

    Intelligent Resume Screening was built as an academic AI/NLP project using Python, Streamlit,
    Sentence Transformers, and scikit-learn.

    **Responsible-use notes**
    - A skill marked “not detected” may still be possessed by the candidate; the term may be absent or phrased differently.
    - Experience and education extraction are heuristic and should be verified against the original resume.
    - Scanned PDFs may need OCR before their text can be analyzed.
    - Scores are evidence indicators, not probabilities of job performance or automatic hiring decisions.
    - Human review is essential; avoid using protected or irrelevant personal characteristics in screening criteria.
    """)

st.divider()
st.markdown("**Intelligent Resume Screening · Developed by Shaiq Hassan**")
st.caption("Local NLP model · No external AI API · Scores summarize text evidence and should support, not replace, human review.")
