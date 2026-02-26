import hashlib
import io
import re
from typing import Any, Dict, List, Set

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI(title="Resume Match API", version="0.1.0")


# -------------------------
# Models
# -------------------------
class MatchRequest(BaseModel):
    resume_id: str
    job_description: str


class AtsRequest(BaseModel):
    resume_id: str


# -------------------------
# Normalization + PDF text
# -------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()


def normalize(text: str) -> str:
    t = (text or "").lower()

    # Remove zero-width / BOM
    t = t.replace("\u200b", "").replace("\ufeff", "")

    # Normalize NBSP
    t = t.replace("\u00a0", " ")

    # Replace bullet-like separators
    t = re.sub(r"[•·●▪■◆▶►]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Collapse spaced letters into words: "s q l" -> "sql"
    t = re.sub(
        r"(?<!\w)(?:[a-z]\s+){1,}[a-z](?!\w)",
        lambda m: m.group(0).replace(" ", ""),
        t,
    )

    # Normalize punctuation spacing
    t = t.replace(" . ", ".").replace(" / ", "/").replace(" - ", "-")

    # Split glued words after known skills (helps pdf extraction issues)
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
        "fastapi",
    ]:
        t = re.sub(rf"({re.escape(skill)})([a-z])", r"\1 \2", t)

    return t


def compact_text(s: str) -> str:
    # Remove whitespace and dots to handle spaced-out emails/urls
    return re.sub(r"[\s\.]", "", (s or "").lower())


def compact_ws(s: str) -> str:
    # remove whitespace only (keep dots) for better email detection
    return re.sub(r"\s+", "", (s or "").lower())


# -------------------------
# Skills (curated + extras)
# -------------------------
# Curated canonical skills and aliases (stable scoring + matching)
SKILL_ALIASES: Dict[str, List[str]] = {
    "c": ["c"],
    "c++": ["c++", "cpp", "c plus plus"],
    "html": ["html"],
    "css": ["css"],
    "streamlit": ["streamlit"],
    "sql": ["sql", "sqlite"],
    "postgresql": ["postgresql", "postgre sql", "postgres", "postgre"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "bash": ["bash", "shell scripting"],
    "powershell": ["powershell", "power shell"],
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "kotlin": ["kotlin"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git"],
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

# Generic “extra skill” extraction (captures unknown tech)
# You can expand this allowlist anytime without breaking API.
GENERIC_TECH_RE = re.compile(
    r"\b("
    r"go|golang|rust|scala|swift|php|ruby|perl|r|matlab|"
    r"graphql|grpc|rest|restapi|microservices|"
    r"firebase|supabase|vercel|netlify|render|railway|"
    r"docker|kubernetes|helm|terraform|ansible|"
    r"nginx|apache|"
    r"redis|kafka|rabbitmq|"
    r"postgresql|mysql|sqlite|mongodb|dynamodb|"
    r"pandas|numpy|scikit[- ]learn|opencv|"
    r"langchain|llamaindex|rag|llm|"
    r"openai|gemini|vertex|huggingface|"
    r"linux|git|github|gitlab|bitbucket"
    r")\b",
    re.I,
)

STOP_EXTRA = set(["rest", "restapi"])  # too generic; remove if you want them listed


def extract_skills(text: str) -> List[str]:
    t = normalize(text)
    found: Set[str] = set()

    for canonical, aliases in SKILL_ALIASES.items():
        for a in aliases:
            pattern = r"(?<!\w)" + re.escape(a) + r"(?!\w)"
            if re.search(pattern, t):
                found.add(canonical)
                break

    return sorted(found)


def extract_extra_skills(text: str, curated: List[str]) -> List[str]:
    t = normalize(text)
    hits = {
        m.group(0).lower().replace(" ", "").replace("-", "")
        for m in GENERIC_TECH_RE.finditer(t)
    }
    # normalize some known tokens
    normalized = set()
    for h in hits:
        if h in STOP_EXTRA:
            continue
        if h == "golang":
            normalized.add("go")
        elif h == "scikitlearn":
            normalized.add("scikit-learn")
        else:
            normalized.add(h)

    # remove anything already covered by curated list
    curated_norm = {c.lower().replace(" ", "").replace("-", "") for c in curated}
    extras = sorted([x for x in normalized if x not in curated_norm])
    return extras[:30]  # keep response sane


# -------------------------
# ATS Readiness (credible)
# -------------------------
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s]+", re.I)
LINK_HINT_RE = re.compile(r"(linkedin\.com|github\.com)", re.I)

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

PERCENT_RE = re.compile(r"\b\d{1,3}(\.\d+)?\s*%\b")
ARROW_RE = re.compile(r"\b\d{1,3}(\.\d+)?\s*%?\s*(->|→)\s*\d{1,3}(\.\d+)?\s*%?\b")

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
DATEISH_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

SECTION_HINTS = {
    "education": ["education", "academic", "university", "college"],
    "skills": ["skills", "technical skills", "tech stack", "tooling"],
    "experience": ["experience", "work experience", "employment", "internship"],
    "projects": ["projects", "project", "personal projects"],
}


def extract_urls(raw: str) -> List[str]:
    urls = URL_RE.findall(raw or "")
    out = []
    for u in urls:
        if u.lower().startswith("www."):
            out.append("https://" + u)
        else:
            out.append(u)
    return out


def has_section(text_norm: str, keys: List[str]) -> bool:
    return any(k in text_norm for k in keys)


def count_achievement_metrics(raw: str) -> int:
    """
    Count only achievement-like metrics.
    Exclude phone numbers, years, and date-like patterns.
    Count:
    - % metrics
    - arrow metrics (82%→90%)
    - numbers on lines containing impact or domain metric words
    - time/unit lines (2 hours, 300ms)
    """
    raw = raw or ""

    scrubbed = PHONE_RE.sub(" ", raw)
    scrubbed = DATEISH_RE.sub(" ", scrubbed)
    scrubbed = YEAR_RE.sub(" ", scrubbed)

    strong = len(PERCENT_RE.findall(scrubbed)) + len(ARROW_RE.findall(scrubbed))

    contextual = 0
    # split on bullets/newlines (works for most resumes)
    lines = []
    if "\n" in scrubbed:
        lines = scrubbed.splitlines()
    elif "•" in scrubbed:
        lines = scrubbed.split("•")
    else:
        lines = scrubbed.split(".")

    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        ln_norm = normalize(ln)
        if not (IMPACT_WORD_RE.search(ln_norm) or DOMAIN_METRIC_RE.search(ln_norm)):
            continue
        if re.search(r"\b\d+(\.\d+)?\b", ln):
            contextual += 1

    unit_hits = 0
    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        if METRIC_UNIT_RE.search(ln) and re.search(r"\b\d+(\.\d+)?\b", ln):
            unit_hits += 1

    # dedupe-ish: contextual and unit_hits can overlap; keep it simple but not huge
    return strong + contextual + max(0, unit_hits - 1)


def ats_readiness_score(raw_text: str, extracted_skills: List[str]) -> dict:
    raw_text = raw_text or ""
    text_norm = normalize(raw_text)
    text_compact = compact_text(raw_text)

    warnings: List[str] = []
    tips: List[str] = []

    # Contact signals
    text_ws = compact_ws(raw_text)

    email_ok = (
        bool(EMAIL_RE.search(raw_text))
        or bool(EMAIL_RE.search(text_ws))
        or bool(re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text_ws))
        or bool(re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text_compact))
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

    # Sections
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

    # Bullets
    bullet_count = sum(raw_text.count(ch) for ch in ["•", "·", "●"]) + raw_text.count(
        "- "
    )
    bullet_pts = 10 if bullet_count >= 6 else (6 if bullet_count >= 2 else 2)
    if bullet_count < 2:
        warnings.append("Very few bullet points detected (ATS readability risk).")
        tips.append("Use bullet points for achievements under projects/experience.")

    # Keyword density
    skill_count = len(set(extracted_skills))
    skill_pts = min(25, skill_count * 3)
    if skill_count < 6:
        warnings.append("Low visible keyword density.")
        tips.append(
            "Add relevant tools/frameworks you actually used (FastAPI, SQL, Git, Docker...)."
        )

    # Metrics / impact (stricter)
    metrics_count = count_achievement_metrics(raw_text)

    if metrics_count >= 4:
        impact_pts = 25
    elif metrics_count >= 2:
        impact_pts = 15
    elif metrics_count >= 1:
        impact_pts = 10
    else:
        impact_pts = 4
        warnings.append("No measurable achievement metrics detected.")
        tips.append(
            "Add numbers: 'reduced latency by 20%', 'improved accuracy 82%→90%', etc."
        )

    impact_hits = len(IMPACT_WORD_RE.findall(text_norm))
    if impact_hits >= 3:
        impact_pts = min(25, impact_pts + 3)

    total = contact_pts + section_pts + bullet_pts + skill_pts + impact_pts
    total = max(0, min(100, int(round(total))))

    # Credibility caps (prevents inflated “perfect” ATS scores)
    if metrics_count < 1:
        total = min(total, 75)
    elif metrics_count < 3:
        total = min(total, 85)

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


# -------------------------
# In-memory store
# -------------------------
RESUMES: Dict[str, Dict[str, Any]] = {}


# -------------------------
# Routes
# -------------------------
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

    curated_skills = extract_skills(text)
    extra_skills = extract_extra_skills(text, curated_skills)

    RESUMES[resume_id] = {
        "resume_hash": resume_hash,
        "text": text,
        "skills": curated_skills,
        "extra_skills": extra_skills,
    }
    return {
        "resume_id": resume_id,
        "skills": curated_skills,
        "extra_skills": extra_skills,
        "text_chars": len(text),
    }


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
        "Quantify impact in your bullets (examples: 'reduced API latency by 20%', "
        "'improved accuracy from 82% to 90%', 'cut build time by 30%')."
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
