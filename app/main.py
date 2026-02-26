import hashlib
import io
import re
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI(title="Resume Match API", version="0.1.0")


class MatchRequest(BaseModel):
    resume_id: str
    job_description: str


# --- ATS Readiness (v1) ---


class AtsRequest(BaseModel):
    resume_id: str


# --- ATS Readiness (v2 fixes) ---

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# Keep phone detection, but we will exclude it from "metrics"
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s]+", re.I)

LINK_HINT_RE = re.compile(r"(linkedin\.com|github\.com)", re.I)

# Words that signal "achievement metric" context
IMPACT_WORD_RE = re.compile(
    r"\b(reduced|improved|increased|decreased|optimized|boosted|cut|saved|grew|raised|lowered)\b",
    re.I,
)
METRIC_UNIT_RE = re.compile(
    r"\b(ms|millisecond|s|sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|day|days|week|weeks|month|months)\b",
    re.I,
)
DOMAIN_METRIC_RE = re.compile(
    r"\b(latency|accuracy|throughput|response time|runtime|memory|cpu|fps|users|requests|qps|tps|errors|cost)\b",
    re.I,
)

# Strong metrics (these we count even without nearby impact words)
PERCENT_RE = re.compile(r"\b\d{1,3}(\.\d+)?\s*%\b")
ARROW_RE = re.compile(r"\b\d{1,3}(\.\d+)?\s*%?\s*(->|→)\s*\d{1,3}(\.\d+)?\s*%?\b")

# Dates/years/phones we should NOT count as achievement metrics
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
DATEISH_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def compact_text(s: str) -> str:
    # remove whitespace and dots to handle "r e h a n u 0 4 @ g m a i l . c o m"
    return re.sub(r"[\s\.]", "", s.lower())


def extract_urls(raw: str) -> list[str]:
    urls = URL_RE.findall(raw or "")
    # normalize "www." to "https://www."
    out = []
    for u in urls:
        if u.lower().startswith("www."):
            out.append("https://" + u)
        else:
            out.append(u)
    return out


def count_achievement_metrics(raw: str, text_norm: str) -> int:
    """
    Count only achievement-like metrics.
    - Count % and arrow patterns
    - Count numbers near impact words or domain metric words
    - Exclude phone numbers, years, and date-like patterns
    """
    raw = raw or ""
    # Remove phone/date/year to reduce false counts
    scrubbed = PHONE_RE.sub(" ", raw)
    scrubbed = DATEISH_RE.sub(" ", scrubbed)
    scrubbed = YEAR_RE.sub(" ", scrubbed)

    # 1) Strong metrics
    strong = len(PERCENT_RE.findall(scrubbed)) + len(ARROW_RE.findall(scrubbed))

    # 2) Contextual metrics: number near impact words or domain metric words
    # Look for patterns like "reduced latency by 20", "improved accuracy to 90%", "saved 2 hours"
    contextual = 0
    lines = scrubbed.splitlines() if "\n" in scrubbed else scrubbed.split("•")
    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        ln_norm = normalize(ln)
        if not (IMPACT_WORD_RE.search(ln_norm) or DOMAIN_METRIC_RE.search(ln_norm)):
            continue
        # number token (1-4 digits) OR decimal, optionally with unit
        if re.search(r"\b\d+(\.\d+)?\b", ln):
            # avoid counting if the only digits are very long ids left over (rare)
            contextual += 1

    # 3) Time/unit metrics without % but with units
    unit_hits = 0
    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        if METRIC_UNIT_RE.search(ln) and re.search(r"\b\d+(\.\d+)?\b", ln):
            unit_hits += 1

    return strong + contextual + unit_hits


def has_section(text_norm: str, keys: list[str]) -> bool:
    return any(k in text_norm for k in keys)


def ats_readiness_score(raw_text: str, extracted_skills: list[str]) -> dict:
    """
    ATS Readiness Score (0-100): parseability + completeness + impact signals.
    Heuristic, not an official ATS score.
    """
    raw_text = raw_text or ""
    text_norm = normalize(raw_text)
    text_compact = compact_text(raw_text)

    warnings = []
    tips = []

    # --- Contact signals (15 pts) ---
    email_ok = (
        bool(EMAIL_RE.search(raw_text))
        or ("@" in text_compact and "gmailcom" in text_compact)
        or ("@gmail" in text_compact)
        or ("@outlook" in text_compact)
    )
    # more robust email check in compact text
    if not email_ok:
        # try compact email regex
        email_ok = bool(
            re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text_compact)
        )

    phone_ok = bool(PHONE_RE.search(raw_text))

    urls = extract_urls(raw_text)
    link_ok = (
        any(LINK_HINT_RE.search(u) for u in urls)
        or ("linkedincom" in text_compact)
        or ("githubcom" in text_compact)
    )

    contact_pts = 0
    if email_ok:
        contact_pts += 5
    else:
        warnings.append("Email not clearly detected.")
        tips.append("Add a visible email in the header (e.g., name@gmail.com).")

    if phone_ok:
        contact_pts += 5
    else:
        warnings.append("Phone number not clearly detected.")
        tips.append("Add a phone number in international format (+91 / +1...).")

    if link_ok:
        contact_pts += 5
    else:
        warnings.append("LinkedIn/GitHub link not detected.")
        tips.append("Add LinkedIn + GitHub links in the header.")

    # --- Section signals (25 pts) ---
    edu_ok = has_section(text_norm, SECTION_HINTS["education"])
    skills_ok = has_section(text_norm, SECTION_HINTS["skills"])
    exp_ok = has_section(text_norm, SECTION_HINTS["experience"])
    proj_ok = has_section(text_norm, SECTION_HINTS["projects"])

    section_pts = 0
    if edu_ok:
        section_pts += 6
    else:
        warnings.append("Education section not clearly detected.")
        tips.append("Add a section heading: 'Education' with degree + college + year.")

    if skills_ok:
        section_pts += 6
    else:
        warnings.append("Skills section not clearly detected.")
        tips.append("Add a section heading: 'Skills' with tech keywords in one place.")

    if exp_ok:
        section_pts += 7
    if proj_ok:
        section_pts += 6

    if not (exp_ok or proj_ok):
        warnings.append("Experience/Projects not clearly detected.")
        tips.append("Add a 'Projects' section with 2–3 projects and bullet points.")

    # --- Bullet/format signals (10 pts) ---
    bullet_count = sum(raw_text.count(ch) for ch in ["•", "·", "●"]) + raw_text.count(
        "- "
    )
    bullet_pts = 10 if bullet_count >= 6 else (6 if bullet_count >= 2 else 2)
    if bullet_count < 2:
        warnings.append("Very few bullet points detected (ATS readability risk).")
        tips.append("Use bullet points for achievements under projects/experience.")

    # --- Keyword density (25 pts) ---
    skill_count = len(set(extracted_skills))
    skill_pts = min(25, skill_count * 3)
    if skill_count < 6:
        warnings.append("Low visible keyword density.")
        tips.append(
            "Add relevant tools/frameworks you actually used (FastAPI, SQL, Git, Docker...)."
        )

    # --- Impact & metrics (25 pts) ---
    metrics_count = count_achievement_metrics(raw_text, text_norm)

    if metrics_count >= 4:
        impact_pts = 25
    elif metrics_count >= 2:
        impact_pts = 18
    elif metrics_count >= 1:
        impact_pts = 12
    else:
        impact_pts = 5
        warnings.append("No measurable achievement metrics detected.")
        tips.append(
            "Add numbers: 'reduced latency by 20%', 'improved accuracy 82%→90%', etc."
        )

    # Small bump if impact verbs appear in the resume text overall
    impact_hits = len(IMPACT_WORD_RE.findall(text_norm))
    if impact_hits >= 3:
        impact_pts = min(25, impact_pts + 4)

    # --- Final score ---
    total = contact_pts + section_pts + bullet_pts + skill_pts + impact_pts
    total = max(0, min(100, int(round(total))))

    label = (
        "Excellent"
        if total >= 85
        else "Strong" if total >= 70 else "Good" if total >= 55 else "Needs improvement"
    )

    return {
        "ats_score": total,
        "label": label,
        "warnings": warnings[:8],
        "tips": tips[:8],
        "signals": {
            "has_email": email_ok,
            "has_phone": phone_ok,
            "has_linkedin_or_github": link_ok,
            "sections": {
                "education": edu_ok,
                "skills": skills_ok,
                "experience": exp_ok,
                "projects": proj_ok,
            },
            "bullet_count": bullet_count,
            "skill_count": skill_count,
            "metrics_count": metrics_count,
        },
    }


# v1 skill list (expand later)
SKILL_ALIASES = {
    "c": ["c"],
    "c++": ["c++", "cpp", "c plus plus"],
    "html": ["html"],
    "css": ["css"],
    "streamlit": ["streamlit"],
    "sql": ["sql", "sqlite", "postgresql", "mysql"],
    "bash": ["bash", "shell scripting"],
    "powershell": ["powershell", "power shell"],
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "kotlin": ["kotlin"],
    "postgresql": ["postgresql", "postgre sql", "postgres", "postgre"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "github": ["github"],
    "linux": ["linux"],
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "django": ["django"],
    "spring": ["spring", "spring boot"],
    "node": ["node", "nodejs", "node.js"],
    "express": ["express", "expressjs"],
    "react": ["react", "reactjs"],
    "next.js": ["next.js", "nextjs", "next js"],
    "android": ["android"],
    "android studio": ["android studio"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "nlp": ["nlp", "natural language processing"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
}

# In-memory store for MVP (Supabase in v2)
RESUMES: Dict[str, Dict[str, Any]] = {}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()


def normalize(text: str) -> str:
    t = text.lower()

    # Remove zero-width / BOM that can break matching
    t = t.replace("\u200b", "").replace("\ufeff", "")

    # Normalize NBSP to normal space
    t = t.replace("\u00a0", " ")

    # Replace common bullet-like separators with spaces
    t = re.sub(r"[•·●▪■◆▶►]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Collapse spaced letters into words: "s q l" -> "sql", "s t r e a m l i t" -> "streamlit"
    t = re.sub(
        r"(?<!\w)(?:[a-z]\s+){1,}[a-z](?!\w)",
        lambda m: m.group(0).replace(" ", ""),
        t,
    )

    # Normalize punctuation spacing
    t = t.replace(" . ", ".").replace(" / ", "/").replace(" - ", "-")

    # Insert a space if a known skill word is immediately followed by letters
    # e.g., "sqlframeworks" -> "sql frameworks", "streamlitconcepts" -> "streamlit concepts"
    for skill in [
        "sql",
        "streamlit",
        "python",
        "docker",
        "github",
        "git",
        "mongodb",
        "nodejs",
        "node.js",
    ]:
        t = re.sub(rf"({re.escape(skill)})([a-z])", r"\1 \2", t)

    return t


def extract_skills(text: str) -> List[str]:
    t = normalize(text)
    found = set()

    for canonical, aliases in SKILL_ALIASES.items():
        for a in aliases:
            # word boundary match where possible
            # handles punctuation like "Python," "Docker/" etc.
            pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
            if re.search(pattern, t):
                found.add(canonical)
                break

    return sorted(found)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    resume_hash = sha256_text(text)
    resume_id = resume_hash[:16]
    skills = extract_skills(text)

    RESUMES[resume_id] = {"resume_hash": resume_hash, "text": text, "skills": skills}
    return {"resume_id": resume_id, "skills": skills, "text_chars": len(text)}


@app.post("/match")
def match(req: MatchRequest):
    if req.resume_id not in RESUMES:
        raise HTTPException(
            status_code=404, detail="resume_id not found. Upload resume first."
        )

    resume_skills = set(RESUMES[req.resume_id]["skills"])
    jd_text = req.job_description or ""
    if len(jd_text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Job description too short.")

    jd_skills = set(extract_skills(jd_text))
    matched = sorted(list(resume_skills.intersection(jd_skills)))
    missing = sorted(list(jd_skills.difference(resume_skills)))

    score = round((len(matched) / max(1, len(jd_skills))) * 100) if jd_skills else 0

    suggestions = []
    if missing:
        suggestions.append(
            f"Highlight/add these (if you have them): {', '.join(missing[:10])}."
        )
    suggestions.append(
        "Quantify impact in your bullets (examples: 'reduced API latency by 20%', 'improved accuracy from 82% to 90%', 'cut build time by 30%')."
    )
    suggestions.append("Add GitHub + deployed links for projects.")

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "jd_skills": sorted(list(jd_skills)),
        "resume_skills": sorted(list(resume_skills)),
        "suggestions": suggestions,
    }


@app.post("/ats")
def ats(req: AtsRequest):
    if req.resume_id not in RESUMES:
        raise HTTPException(
            status_code=404, detail="resume_id not found. Upload resume first."
        )

    raw_text = RESUMES[req.resume_id]["text"]
    skills = RESUMES[req.resume_id].get("skills", [])
    return ats_readiness_score(raw_text, skills)
