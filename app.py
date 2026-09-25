from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import re


app = Flask(__name__)


# ==========================================================
# JOB DATABASE
# ==========================================================

JOBS = [
    {
        "title": "Python Developer",
        "description": """
        Python Flask Django REST API SQL Git GitHub
        backend development database programming
        object oriented programming debugging
        """
    },

    {
        "title": "Data Analyst",
        "description": """
        Python SQL Excel Power BI Tableau Pandas NumPy
        data analysis statistics data visualization
        dashboards reporting data cleaning
        """
    },

    {
        "title": "Machine Learning Engineer",
        "description": """
        Python Machine Learning Scikit-learn TensorFlow
        Pandas NumPy NLP model training data preprocessing
        classification regression deep learning
        """
    },

    {
        "title": "Web Developer",
        "description": """
        HTML CSS JavaScript Bootstrap React
        Python Flask web development frontend backend
        responsive design REST API Git
        """
    },

    {
        "title": "AI / NLP Engineer",
        "description": """
        Python NLP Natural Language Processing
        Machine Learning Scikit-learn NLTK Transformers
        text classification sentiment analysis
        TF-IDF cosine similarity language models
        """
    }
]


# ==========================================================
# SKILLS DATABASE
# ==========================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "html",
    "css",
    "javascript",
    "react",
    "bootstrap",
    "flask",
    "django",
    "rest api",
    "git",
    "github",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "nltk",
    "transformers",
    "data analysis",
    "statistics",
    "data visualization",
    "mysql",
    "mongodb"
]


# ==========================================================
# COURSE / LEARNING RESOURCE DATABASE
# ==========================================================

COURSES = {
    "python": {
        "course": "Python Programming",
        "provider": "Python Official Tutorial",
        "url": "https://docs.python.org/3/tutorial/"
    },

    "sql": {
        "course": "SQL Fundamentals",
        "provider": "Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/training/paths/get-started-querying-with-transact-sql/"
    },

    "power bi": {
        "course": "Power BI Fundamentals",
        "provider": "Microsoft Learn",
        "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/"
    },

    "tableau": {
        "course": "Tableau Learning",
        "provider": "Tableau",
        "url": "https://www.tableau.com/learn/training"
    },

    "machine learning": {
        "course": "Machine Learning Fundamentals",
        "provider": "Google Machine Learning",
        "url": "https://developers.google.com/machine-learning/crash-course"
    },

    "scikit-learn": {
        "course": "Scikit-learn User Guide",
        "provider": "Scikit-learn",
        "url": "https://scikit-learn.org/stable/user_guide.html"
    },

    "nlp": {
        "course": "Natural Language Processing",
        "provider": "Hugging Face",
        "url": "https://huggingface.co/learn/nlp-course"
    },

    "git": {
        "course": "Git and GitHub",
        "provider": "GitHub Skills",
        "url": "https://skills.github.com/"
    },

    "javascript": {
        "course": "JavaScript Fundamentals",
        "provider": "MDN Web Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"
    },

    "react": {
        "course": "React Fundamentals",
        "provider": "React",
        "url": "https://react.dev/learn"
    },

    "html": {
        "course": "HTML Fundamentals",
        "provider": "MDN Web Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Structuring_content"
    },

    "css": {
        "course": "CSS Fundamentals",
        "provider": "MDN Web Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Styling_basics"
    },

    "excel": {
        "course": "Microsoft Excel Training",
        "provider": "Microsoft Support",
        "url": "https://support.microsoft.com/en-us/excel"
    },

    "data analysis": {
        "course": "Data Analysis with Python",
        "provider": "freeCodeCamp",
        "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/"
    },

    "statistics": {
        "course": "Statistics Fundamentals",
        "provider": "Khan Academy",
        "url": "https://www.khanacademy.org/math/statistics-probability"
    }
}


# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# SKILL EXTRACTION
# ==========================================================

def extract_skills(resume_text):

    resume_text = clean_text(resume_text)

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, resume_text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# ==========================================================
# TF-IDF + COSINE SIMILARITY
# ==========================================================

def calculate_match(
    resume_text,
    job_description
):

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(
        similarity * 100,
        2
    )


# ==========================================================
# MISSING SKILLS
# ==========================================================

def get_missing_skills(
    resume_text,
    job_description
):

    resume_text = clean_text(
        resume_text
    )

    required_skills = []

    for skill in SKILLS:

        pattern = (
            r"\b"
            + re.escape(skill.lower())
            + r"\b"
        )

        if re.search(
            pattern,
            job_description.lower()
        ):

            required_skills.append(skill)


    missing_skills = []

    for skill in required_skills:

        pattern = (
            r"\b"
            + re.escape(skill.lower())
            + r"\b"
        )

        if not re.search(
            pattern,
            resume_text
        ):

            missing_skills.append(skill)

    return sorted(
        set(missing_skills)
    )


# ==========================================================
# PROFILE SCORE
# ==========================================================

def calculate_profile_score(text):

    score = 0

    text_lower = text.lower()


    # Email
    if re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    ):

        score += 25


    # Phone
    if re.search(
        r"\+?\d[\d\s\-()]{8,}\d",
        text
    ):

        score += 25


    # LinkedIn
    if "linkedin" in text_lower:

        score += 25


    # GitHub
    if "github" in text_lower:

        score += 25


    return score


# ==========================================================
# SKILLS SCORE
# ==========================================================

def calculate_skills_score(
    extracted_skills,
    results
):

    if not extracted_skills:

        return 0


    # Maximum 60 points for detected skills

    skill_count_score = min(
        len(extracted_skills) * 8,
        60
    )


    # Job relevance contributes 40 points

    relevance_score = 0

    if results:

        relevance_score = (
            results[0]["match"] * 0.40
        )


    total = (
        skill_count_score
        + relevance_score
    )


    return round(
        min(total, 100),
        2
    )


# ==========================================================
# EDUCATION SCORE
# ==========================================================

def calculate_education_score(text):

    text_lower = text.lower()

    education_keywords = [

        "education",
        "b.tech",
        "btech",
        "b.e",
        "be ",
        "m.tech",
        "mtech",
        "mca",
        "bca",
        "b.sc",
        "bsc",
        "m.sc",
        "msc",
        "bachelor",
        "master",
        "degree",
        "university",
        "college"

    ]


    found = 0


    for keyword in education_keywords:

        if keyword in text_lower:

            found += 1


    if found == 0:

        return 0


    if found <= 2:

        return 50


    if found <= 4:

        return 75


    return 100


# ==========================================================
# EXPERIENCE SCORE
# ==========================================================

def calculate_experience_score(text):

    text_lower = text.lower()

    experience_keywords = [

        "experience",
        "work experience",
        "internship",
        "intern",
        "developer",
        "analyst",
        "engineer",
        "worked",
        "employment",
        "professional experience"

    ]


    found = 0


    for keyword in experience_keywords:

        if keyword in text_lower:

            found += 1


    if found == 0:

        return 30


    if found <= 2:

        return 60


    if found <= 4:

        return 80


    return 100


# ==========================================================
# PROJECT SCORE
# ==========================================================

def calculate_projects_score(text):

    text_lower = text.lower()

    project_keywords = [

        "project",
        "projects",
        "developed",
        "built",
        "created",
        "implemented",
        "application",
        "system"

    ]


    found = 0


    for keyword in project_keywords:

        if keyword in text_lower:

            found += 1


    if found == 0:

        return 0


    if found <= 2:

        return 50


    if found <= 4:

        return 75


    return 100


# ==========================================================
# FORMATTING SCORE
# ==========================================================

def calculate_formatting_score(text):

    score = 0

    text_lower = text.lower()


    important_sections = [

        "education",
        "skills",
        "experience",
        "projects"

    ]


    for section in important_sections:

        if section in text_lower:

            score += 20


    word_count = len(
        text.split()
    )


    if 150 <= word_count <= 1200:

        score += 20

    elif word_count > 50:

        score += 10


    return min(
        score,
        100
    )


# ==========================================================
# SUGGESTIONS
# ==========================================================

def generate_suggestions(

    profile_score,

    skills_score,

    education_score,

    experience_score,

    projects_score,

    formatting_score,

    missing_skills

):

    suggestions = []


    if profile_score < 75:

        suggestions.append(
            "Add complete contact information including email, phone, LinkedIn and GitHub."
        )


    if skills_score < 70:

        suggestions.append(
            "Add more relevant technical skills and mention the technologies used in your projects."
        )


    if education_score < 70:

        suggestions.append(
            "Add a clearly structured Education section with degree, college or university and graduation details."
        )


    if experience_score < 70:

        suggestions.append(
            "Add internship, work experience or practical experience. Describe your responsibilities and achievements."
        )


    if projects_score < 70:

        suggestions.append(
            "Add 2-3 strong technical projects and explain the technologies, features and results."
        )


    if formatting_score < 70:

        suggestions.append(
            "Improve resume structure by clearly separating Skills, Education, Experience and Projects sections."
        )


    if missing_skills:

        skills_text = ", ".join(
            missing_skills[:5]
        )

        suggestions.append(
            f"Consider learning these missing job-related skills: {skills_text}."
        )


    if not suggestions:

        suggestions.append(
            "Your resume has good coverage. Continue improving it with measurable achievements and relevant projects."
        )


    return suggestions


# ==========================================================
# COURSE RECOMMENDATIONS
# ==========================================================

def recommend_courses(
    missing_skills
):

    recommendations = []


    for skill in missing_skills:

        if skill in COURSES:

            recommendations.append({

                "skill": skill,

                "course": COURSES[skill]["course"],

                "provider": COURSES[skill]["provider"],

                "url": COURSES[skill]["url"]

            })


    return recommendations[:8]


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    results = []

    resume_text = ""

    extracted_skills = []

    uploaded_filename = ""

    error = ""

    best_job = None

    best_match = 0

    overall_score = 0

    section_scores = {}

    suggestions = []

    course_recommendations = []

    missing_skills = []


    # ======================================================
    # POST REQUEST
    # ======================================================

    if request.method == "POST":

        resume_file = request.files.get(
            "resume_file"
        )

        typed_resume = request.form.get(
            "resume_text",
            ""
        ).strip()


        # ==================================================
        # PDF UPLOAD
        # ==================================================

        if resume_file and resume_file.filename:

            if resume_file.filename.lower().endswith(
                ".pdf"
            ):

                try:

                    resume_text = extract_pdf_text(
                        resume_file
                    )

                    uploaded_filename = (
                        resume_file.filename
                    )

                except Exception:

                    error = (
                        "Could not read the PDF file."
                    )

            else:

                error = (
                    "Please upload a PDF file."
                )


        # ==================================================
        # PASTED TEXT
        # ==================================================

        elif typed_resume:

            resume_text = typed_resume


        else:

            error = (
                "Please upload a PDF resume "
                "or paste your resume text."
            )


        # ==================================================
        # ANALYZE RESUME
        # ==================================================

        if resume_text and not error:


            # ------------------------------------------------
            # SKILLS
            # ------------------------------------------------

            extracted_skills = extract_skills(
                resume_text
            )


            # ------------------------------------------------
            # JOB MATCHING
            # ------------------------------------------------

            for job in JOBS:

                match_percentage = calculate_match(

                    resume_text,

                    job["description"]

                )


                job_missing_skills = get_missing_skills(

                    resume_text,

                    job["description"]

                )


                results.append({

                    "title": job["title"],

                    "match": match_percentage,

                    "missing_skills": job_missing_skills

                })


            # ------------------------------------------------
            # SORT JOB RESULTS
            # ------------------------------------------------

            results.sort(

                key=lambda x: x["match"],

                reverse=True

            )


            # ------------------------------------------------
            # BEST JOB
            # ------------------------------------------------

            if results:

                best_job = results[0]["title"]

                best_match = results[0]["match"]

                missing_skills = results[0][
                    "missing_skills"
                ]


            # ==================================================
            # SECTION SCORES
            # ==================================================

            profile_score = calculate_profile_score(
                resume_text
            )


            skills_score = calculate_skills_score(

                extracted_skills,

                results

            )


            education_score = calculate_education_score(
                resume_text
            )


            experience_score = calculate_experience_score(
                resume_text
            )


            projects_score = calculate_projects_score(
                resume_text
            )


            formatting_score = calculate_formatting_score(
                resume_text
            )


            # ==================================================
            # SECTION SCORE DICTIONARY
            # ==================================================

            section_scores = {

                "Profile": profile_score,

                "Skills": skills_score,

                "Education": education_score,

                "Experience": experience_score,

                "Projects": projects_score,

                "Formatting": formatting_score,

                "Job Match": best_match

            }


            # ==================================================
            # OVERALL RESUME SCORE
            # ==================================================

            overall_score = round(

                (

                    profile_score * 0.10

                    + skills_score * 0.20

                    + education_score * 0.10

                    + experience_score * 0.15

                    + projects_score * 0.15

                    + formatting_score * 0.10

                    + best_match * 0.20

                ),

                2

            )


            # ==================================================
            # SUGGESTIONS
            # ==================================================

            suggestions = generate_suggestions(

                profile_score,

                skills_score,

                education_score,

                experience_score,

                projects_score,

                formatting_score,

                missing_skills

            )


            # ==================================================
            # COURSE RECOMMENDATIONS
            # ==================================================

            course_recommendations = recommend_courses(

                missing_skills

            )


    # ======================================================
    # SEND DATA TO HTML
    # ======================================================

    return render_template(

        "index.html",

        results=results,

        resume_text=resume_text,

        extracted_skills=extracted_skills,

        uploaded_filename=uploaded_filename,

        error=error,

        best_job=best_job,

        best_match=best_match,

        overall_score=overall_score,

        section_scores=section_scores,

        suggestions=suggestions,

        course_recommendations=course_recommendations,

        missing_skills=missing_skills

    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )