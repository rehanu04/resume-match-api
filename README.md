# ResumeMatch API (FastAPI)

A lightweight resume analyzer + job matcher API.

It supports:
- **ATS Readiness Score** (how parseable / complete / ATS-friendly the resume is)
- **Job Match Score** (how well the resume matches a given job description)
- **Skill extraction** from resume + JD (keyword-based, fast, no paid APIs)

## Live Demo (Deployed)
- Swagger UI: https://resume-match-api-gace.onrender.com/docs  
- Health: https://resume-match-api-gace.onrender.com/health
  
Android app repo: https://github.com/rehanu04/resumematch-android

Android Release download: https://github.com/rehanu04/resumematch-android/releases
## Endpoints
### 1) Upload Resume (PDF)
`POST /resume/upload`  
**Input:** multipart file (`.pdf`)  
**Output:** `resume_id`, extracted `skills`, `extra_skills`, `text_chars`

### 2) ATS Readiness
`POST /ats`  
**Input:** `{ "resume_id": "<id>" }`  
**Output:** `ats_score (0-100)`, `label`, `warnings`, `tips`, plus debug `signals`

> Note: “ATS Readiness” is a **heuristic score** (not an official ATS score).  
It estimates how ATS-friendly the resume is based on structure, contact detection, sections, keyword density, and measurable metrics.

### 3) Job Match
`POST /match`  
**Input:**  
```json
{ "resume_id": "<id>", "job_description": "..." }

