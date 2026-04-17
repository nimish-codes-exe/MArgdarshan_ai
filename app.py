import streamlit as st
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import re

# PDF processing
try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Margdarshak AI - Career Guide for Tier 2/3 Students",
    page_icon="🎯",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>

/* ================= GLOBAL ================= */
.stApp {
    background: linear-gradient(135deg, #DBEAFE 0%, #F8FAFC 100%);
    color: #1F2937;
    font-family: 'Segoe UI', sans-serif;
}

/* ================= GLASS EFFECT ================= */
.glass {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0 10px 35px rgba(0,0,0,0.08);
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ================= HEADER ================= */
.main-header {
    background: linear-gradient(135deg, rgba(37,99,235,0.9), rgba(30,58,138,0.9));
    backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(37,99,235,0.3);
    margin-bottom: 2rem;
}

.main-header h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
}

.main-header p {
    color: #DBEAFE;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.hindi-tagline {
    color: #93C5FD;
    font-size: 1.3rem;
}

/* ================= BUTTONS ================= */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1E40AF);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(37,99,235,0.4);
}

/* ================= PROJECT CARDS ================= */
.project-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.7rem 0;
    transition: all 0.25s ease;
    border-left: 4px solid #2563EB;
}

.project-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}

/* ================= SKILL TAGS ================= */
.skill-match {
    background: #DCFCE7;
    color: #166534;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.85rem;
}

.skill-missing {
    background: #FEE2E2;
    color: #991B1B;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.85rem;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
    border-right: 1px solid #E5E7EB;
}

/* ================= INPUT FIELDS ================= */
textarea, input {
    border-radius: 8px !important;
    border: 1px solid #D1D5DB !important;
}

/* ================= JOB CARDS ================= */
.job-card {
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
    font-weight: 600;
    transition: all 0.25s ease;
}

.job-card:hover {
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 12px 25px rgba(0,0,0,0.2);
}

/* ================= EXPANDER HOVER ================= */
details {
    transition: all 0.2s ease;
}

details:hover {
    transform: scale(1.01);
}

/* ================= METRICS (SUBTLE ANIMATION) ================= */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.6);
    padding: 10px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
}

/* ================= SCROLLBAR (BONUS CLEAN LOOK) ================= */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #CBD5F5;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# PDF TEXT EXTRACTION
# ============================================
def extract_text_from_pdf(pdf_file):
    if not PDF_AVAILABLE:
        return "PDF support not available. Install: pip install PyPDF2"
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# ============================================
# RESUME EVALUATION
# ============================================
def evaluate_resume(resume_text, target_role):
    role_requirements = {
        "Data Analyst": {
            "must_have": ["sql", "excel", "python", "data analysis", "statistics"],
            "good_to_have": ["tableau", "power bi", "r", "pandas", "numpy"],
            "soft_skills": ["communication", "problem solving", "analytical thinking"]
        },
        "Frontend Developer": {
            "must_have": ["html", "css", "javascript", "react"],
            "good_to_have": ["typescript", "angular", "vue", "next.js", "tailwind"],
            "soft_skills": ["creativity", "attention to detail", "teamwork"]
        },
        "Backend Developer": {
            "must_have": ["python", "sql", "api", "database"],
            "good_to_have": ["django", "node.js", "mongodb", "postgresql", "docker"],
            "soft_skills": ["problem solving", "logical thinking", "debugging"]
        },
        "ML Engineer": {
            "must_have": ["python", "machine learning", "statistics", "pandas"],
            "good_to_have": ["tensorflow", "pytorch", "deep learning", "nlp"],
            "soft_skills": ["mathematical thinking", "research", "curiosity"]
        }
    }

    matched_role = "Data Analyst"
    for role in role_requirements:
        if role.lower() in target_role.lower():
            matched_role = role
            break

    requirements = role_requirements[matched_role]
    resume_lower = resume_text.lower()

    found_must = [s for s in requirements["must_have"] if s in resume_lower]
    found_good = [s for s in requirements["good_to_have"] if s in resume_lower]
    found_soft = [s for s in requirements["soft_skills"] if s in resume_lower]

    missing_must = [s for s in requirements["must_have"] if s not in resume_lower]
    missing_good = [s for s in requirements["good_to_have"] if s not in resume_lower]

    must_score = (len(found_must) / len(requirements["must_have"])) * 100 if requirements["must_have"] else 100
    good_score = (len(found_good) / len(requirements["good_to_have"])) * 100 if requirements["good_to_have"] else 100
    soft_score = (len(found_soft) / len(requirements["soft_skills"])) * 100 if requirements["soft_skills"] else 100

    total_score = (must_score * 0.5 + good_score * 0.3 + soft_score * 0.2)
    word_count = len(resume_lower.split())
    if word_count < 200:
        total_score = total_score * 0.8

    total_score = min(100, max(0, round(total_score, 1)))

    recommendations = []
    if total_score < 50:
        recommendations.append("🔴 Critical: Resume needs improvement")
    elif total_score < 70:
        recommendations.append("🟡 Moderate: Shows potential")
    else:
        recommendations.append("🟢 Good: Competitive resume")

    if missing_must:
        recommendations.append(f"📚 Must-Learn: {', '.join(missing_must[:5])}")

    return {
        "total_score": total_score,
        "must_have_score": round(must_score, 1),
        "good_to_have_score": round(good_score, 1),
        "soft_skills_score": round(soft_score, 1),
        "found_must_have": found_must,
        "found_good_to_have": found_good,
        "found_soft_skills": found_soft,
        "missing_must_have": missing_must,
        "missing_good_to_have": missing_good,
        "matched_role": matched_role,
        "word_count": word_count,
        "recommendations": recommendations
    }


# ============================================
# JOB SEARCH URLS
# ============================================
def get_naukri_url(skill):
    return f"https://www.naukri.com/{skill.lower().replace(' ', '-')}-jobs?experience=0"


def get_linkedin_url(skill):
    return f"https://www.linkedin.com/jobs/search?keywords={skill.replace(' ', '%20')}&f_E=2&location=India"


# ============================================
# PROJECT SUGGESTIONS
# ============================================
def get_project_suggestions(role, skills):
    projects = {
        "Data Analyst": [
            {"name": "📊 Sales Dashboard", "description": "Build with Python, Pandas, Power BI",
             "skills": ["Python", "Pandas", "SQL"], "difficulty": "Beginner", "timeline": "2-3 weeks"},
            {"name": "📈 Churn Prediction", "description": "Logistic regression model",
             "skills": ["Python", "Scikit-learn"], "difficulty": "Intermediate", "timeline": "3-4 weeks"}
        ],
        "Frontend": [
            {"name": "🎬 Netflix Clone", "description": "Movie app with React and API",
             "skills": ["React", "API", "CSS"], "difficulty": "Intermediate", "timeline": "3-4 weeks"},
            {"name": "🛒 E-commerce Gallery", "description": "Product catalog with cart",
             "skills": ["JavaScript", "HTML/CSS"], "difficulty": "Beginner", "timeline": "2-3 weeks"}
        ],
        "Backend": [
            {"name": "🔐 Auth System", "description": "JWT authentication", "skills": ["Node.js", "Express", "MongoDB"],
             "difficulty": "Intermediate", "timeline": "2-3 weeks"},
            {"name": "📦 E-commerce API", "description": "REST API for e-commerce", "skills": ["Django", "PostgreSQL"],
             "difficulty": "Advanced", "timeline": "4-5 weeks"}
        ],
        "ML": [
            {"name": "🖼️ Image Classifier", "description": "CNN with TensorFlow", "skills": ["Python", "TensorFlow"],
             "difficulty": "Intermediate", "timeline": "3-4 weeks"},
            {"name": "📝 Sentiment Analysis", "description": "NLP classification", "skills": ["Python", "NLP"],
             "difficulty": "Intermediate", "timeline": "3-4 weeks"}
        ]
    }
    for key in projects:
        if key.lower() in role.lower():
            return projects[key]
    return [
        {"name": "📁 Portfolio Website", "description": "Personal portfolio", "skills": ["HTML/CSS", "JavaScript"],
         "difficulty": "Beginner", "timeline": "1-2 weeks"},
        {"name": "✅ Task Manager", "description": "Full-stack app", "skills": ["Any Backend", "SQL"],
         "difficulty": "Intermediate", "timeline": "3-4 weeks"}
    ]


# ============================================
# CREATE DATASET
# ============================================
if not os.path.exists("job_data.csv"):
    st.info("🔄 Creating job dataset...")
    roles = ["Frontend Developer", "Backend Developer", "Full Stack Developer", "Data Analyst", "Data Scientist",
             "ML Engineer", "DevOps Engineer", "Cloud Engineer"]
    skills_pool = {
        "Frontend Developer": ["React", "JavaScript", "HTML5", "CSS3", "TypeScript"],
        "Backend Developer": ["Python", "Django", "Node.js", "SQL", "MongoDB"],
        "Full Stack Developer": ["React", "Node.js", "MongoDB", "Python", "AWS"],
        "Data Analyst": ["SQL", "Excel", "Python", "Tableau", "Power BI"],
        "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL", "Pandas"],
        "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Scikit-learn"],
        "DevOps Engineer": ["Docker", "Kubernetes", "Jenkins", "AWS", "Linux"],
        "Cloud Engineer": ["AWS", "Azure", "Terraform", "Kubernetes", "Docker"]
    }
    companies = ["Google", "Microsoft", "Amazon", "TCS", "Infosys", "Wipro", "Accenture", "Zomato", "Flipkart"]
    locations = ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Chennai", "Noida", "Remote"]
    jobs = []
    for i in range(300):
        role = random.choice(roles)
        skills = skills_pool.get(role, ["Python", "SQL", "Git"])
        jobs.append({
            "job_id": i,
            "title": f"{random.choice(['Junior', 'Associate', 'Fresher'])} {role}".strip(),
            "company": random.choice(companies),
            "required_skills": ", ".join(random.sample(skills, k=min(4, len(skills)))),
            "location": random.choice(locations),
            "salary_range": f"₹{random.randint(3, 8)}-{random.randint(8, 20)} LPA"
        })
    df = pd.DataFrame(jobs)
    df.to_csv("job_data.csv", index=False)
    st.success(f"✅ Created {len(df)} job listings!")


# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv("job_data.csv")


df = load_data()


# ============================================
# TF-IDF VECTORIZER
# ============================================
@st.cache_resource
def create_vectorizer():
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    skill_matrix = vectorizer.fit_transform(df['required_skills'])
    return vectorizer, skill_matrix


vectorizer, skill_matrix = create_vectorizer()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 📝 Your Profile")
    default_skills = st.session_state.get('auto_skills', '')
    user_skills = st.text_area("Enter your skills:", value=default_skills, placeholder="e.g., Python, SQL, Excel, ML")

    col1, col2 = st.columns(2)
    with col1:
        tier = st.selectbox("College tier:", ["Tier 2", "Tier 3"])
    with col2:
        year = st.selectbox("Year:", ["1st Year", "2nd Year", "3rd Year", "Final Year", "Graduated"])

    search_button = st.button("🔍 Find My Career Path", type="primary", use_container_width=True)

    # Resume Evaluation
    st.divider()
    st.subheader("📄 Resume Evaluation")
    uploaded_file = st.file_uploader("Upload Resume (TXT or PDF)", type=['txt', 'pdf'])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
            if "Error" in resume_text or "not available" in resume_text:
                st.error(resume_text)
            else:
                st.success("✅ PDF uploaded!")
        else:
            resume_text = uploaded_file.read().decode()
        st.session_state['resume_text'] = resume_text

        eval_role = st.selectbox("Target Role:",
                                 ["Data Analyst", "Frontend Developer", "Backend Developer", "ML Engineer"])

        if st.button("📊 Evaluate My Resume", use_container_width=True):
            st.session_state['eval_role'] = eval_role
            st.session_state['show_evaluation'] = True
            st.rerun()

        common_skills = ['python', 'java', 'sql', 'excel', 'javascript', 'react', 'node', 'aws', 'docker']
        found_skills = [skill for skill in common_skills if skill in resume_text.lower()]
        if found_skills:
            st.success(f"✅ Found: {', '.join(found_skills[:8])}")
            if st.button("📎 Use These Skills"):
                st.session_state['auto_skills'] = ', '.join(found_skills)
                st.rerun()

    # Premium Pathway
    st.divider()
    st.subheader("🏛️ Premium Pathway")
    if st.checkbox("Enable Advanced Counseling"):
        advice = {
            "Tier 3": {"1st Year": "🎯 DSA + GitHub. 100 days of code.", "2nd Year": "💡 Pick 1 domain.",
                       "3rd Year": "🚀 Build projects.", "Final Year": "📈 Get referrals."},
            "Tier 2": {"1st Year": "🌟 CP + Build in public.", "2nd Year": "🔗 Meetups.", "3rd Year": "💼 LeetCode.",
                       "Final Year": "🎯 Multiple offers."}
        }
        st.info(advice[tier].get(year, "Focus on building projects!"))

    st.divider()
    st.caption("© 2024 Margdarshak AI")

# ============================================
# DISPLAY RESUME EVALUATION
# ============================================
if st.session_state.get('show_evaluation', False):
    st.divider()
    st.subheader("📊 Resume Evaluation Results")
    result = evaluate_resume(st.session_state['resume_text'], st.session_state['eval_role'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall", f"{result['total_score']}%")
    with col2:
        st.metric("Must-Have", f"{result['must_have_score']}%")
    with col3:
        st.metric("Good-to-Have", f"{result['good_to_have_score']}%")
    with col4:
        st.metric("Soft Skills", f"{result['soft_skills_score']}%")

    st.progress(result['total_score'] / 100)

    st.markdown("### ✅ Skills Found")
    col1, col2, col3 = st.columns(3)
    with col1:
        for skill in result['found_must_have']:
            st.markdown(f'<span class="skill-match">{skill}</span>', unsafe_allow_html=True)
    with col2:
        for skill in result['found_good_to_have']:
            st.markdown(f'<span class="skill-match">{skill}</span>', unsafe_allow_html=True)
    with col3:
        for skill in result['found_soft_skills']:
            st.markdown(f'<span class="skill-match">{skill}</span>', unsafe_allow_html=True)

    if result['missing_must_have']:
        st.warning(f"📚 Must-Learn: {', '.join(result['missing_must_have'][:5])}")

    for rec in result['recommendations']:
        st.markdown(f"- {rec}")

    if st.button("Clear Evaluation"):
        st.session_state['show_evaluation'] = False
        st.rerun()
    st.divider()

# ============================================
# MAIN LOGIC - SEARCH
# ============================================
if search_button and user_skills:
    user_vector = vectorizer.transform([user_skills])
    similarities = cosine_similarity(user_vector, skill_matrix).flatten()
    df['similarity_score'] = similarities

    available_columns = ['title', 'company', 'required_skills', 'similarity_score', 'location']
    if 'salary_range' in df.columns:
        available_columns.append('salary_range')

    top_matches = df.nlargest(10, 'similarity_score')[available_columns]

    st.subheader(f"🎯 Top Career Matches")
    st.caption(f"Based on: *{user_skills}*")

    col1, col2 = st.columns([2, 1])

    with col1:
        for idx, row in top_matches.head(5).iterrows():
            with st.expander(f"**{row['title']}** at {row['company']} - {row['similarity_score'] * 100:.1f}% match"):
                st.write(f"📍 {row['location']}")
                if 'salary_range' in row.index:
                    st.write(f"💰 {row['salary_range']}")
                st.write(f"**Skills:** {row['required_skills']}")

                user_set = set(s.strip().lower() for s in user_skills.split(','))
                job_set = set(s.strip().lower() for s in row['required_skills'].split(','))
                missing = job_set - user_set
                matching = job_set & user_set

                st.success(f"✅ {', '.join(matching) if matching else 'None'}")
                if missing:
                    st.warning(f"📚 Learn: {', '.join(missing)}")

    with col2:
        st.subheader("📊 In-Demand Skills")
        all_skills = [s.strip().lower() for skills in top_matches['required_skills'] for s in skills.split(',')]
        if all_skills:
            st.bar_chart(pd.Series(all_skills).value_counts().head(8))

    # Project Suggestions
    st.divider()
    st.subheader("🚀 Portfolio Projects")
    target_role = top_matches.iloc[0]['title']
    projects = get_project_suggestions(target_role, user_skills)

    cols = st.columns(2)
    for i, proj in enumerate(projects[:2]):
        with cols[i]:
            st.markdown(f"""
            <div class="project-card">
                <h4>{proj['name']}</h4>
                <p><small>{proj['description']}</small></p>
                <p>🛠️ Skills: {', '.join(proj['skills'][:3])}</p>
                <p>📊 {proj['difficulty']} | ⏱️ {proj['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Job Search Links
    st.divider()
    st.subheader("🔗 Direct Job Search")
    primary_skill = user_skills.split(',')[0].strip()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <a href="{get_naukri_url(primary_skill)}" target="_blank">
            <div style="background: #FF5722; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>🔍 Naukri.com</h3>
                <p>Search '{primary_skill}' jobs</p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <a href="{get_linkedin_url(primary_skill)}" target="_blank">
            <div style="background: #0077B5; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>💼 LinkedIn</h3>
                <p>Search '{primary_skill}' jobs</p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <a href="https://www.foundit.in/search?q={primary_skill}" target="_blank">
            <div style="background: #667eea; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3>🎯 Foundit</h3>
                <p>Search '{primary_skill}' jobs</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

elif search_button:
    st.warning("⚠️ Please enter your skills")


# ============================================
# WELCOME SCREEN
# ============================================
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 👋 Welcome to Margdarshak AI!

        **Your AI-powered career guide for Tier 2/3 engineering students.**

        **Features:**
        - 📝 Skill-based job matching
        - 📄 Resume evaluation with ATS score
        - 🚀 Personalized project suggestions
        - 🔗 Direct job search links (Naukri, LinkedIn, Foundit)
        - 🏛️ Premium career counseling

        *"From a tier 3 college to your dream job - we'll show you the path."*
        """)

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 Sample Jobs in Database")
        st.caption("300+ jobs across multiple roles with salary insights")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("📊 Platform Stats")
        st.metric("📊 Total Jobs", len(df))
        st.metric("🏢 Companies", df['company'].nunique())
        st.metric("📍 Cities", df['location'].nunique())

        st.divider()
        st.subheader("💡 Quick Tips")
        st.info("""
        **For Best Results:**
        1. Enter 5-10 skills separated by commas
        2. Upload your resume (TXT or PDF) for evaluation
        3. Click on job matches to see skill gaps
        4. Use direct job links to search live openings
        """)

        st.subheader("📊 In-Demand Skills")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<style>

/* ================= JOB CARD ANIMATIONS ================= */

/* ENTRY ANIMATION */
@keyframes fadeSlideUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* APPLY TO JOB CARDS */
.job-card {
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
    font-weight: 600;
    transition: all 0.25s ease;

    /* animation */
    animation: fadeSlideUp 0.6s ease forwards;
}

/* STAGGER EFFECT (delay for each card) */
.job-card:nth-child(1) { animation-delay: 0.1s; }
.job-card:nth-child(2) { animation-delay: 0.2s; }
.job-card:nth-child(3) { animation-delay: 0.3s; }

/* HOVER EFFECT */
.job-card:hover {
    transform: translateY(-6px) scale(1.04);
    box-shadow: 0 14px 30px rgba(0,0,0,0.25);
}

/* CLICK EFFECT */
.job-card:active {
    transform: scale(0.97);
}

</style>
""", unsafe_allow_html=True)