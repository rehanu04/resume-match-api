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
        "Quantify impact: reduced latency by X%, improved accuracy by Y%."
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


@app.get("/resume/{resume_id}/preview")
def resume_preview(resume_id: str):
    if resume_id not in RESUMES:
        raise HTTPException(status_code=404, detail="resume_id not found")
    text = RESUMES[resume_id]["text"]
    # show first 1200 chars for debugging
    return {"resume_id": resume_id, "preview": text[:1200]}


@app.get("/resume/{resume_id}/preview_norm")
def resume_preview_norm(resume_id: str):
    if resume_id not in RESUMES:
        raise HTTPException(status_code=404, detail="resume_id not found")
    raw = RESUMES[resume_id]["text"]
    norm = normalize(raw)
    return {
        "resume_id": resume_id,
        "has_sql": ("sql" in norm),
        "has_streamlit": ("streamlit" in norm),
        "preview_norm": norm[:1200],
    }
