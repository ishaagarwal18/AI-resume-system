import time
from typing import List
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.agents.parser_agent import ParserAgent
from app.agents.normalization_agent import NormalizationAgent
from app.agents.matcher_agent import MatcherAgent
from app.orchestrator import Orchestrator

router = APIRouter()


# Request model
class MatchRequest(BaseModel):
    resume_text: str
    job_skills: list[str]


@router.get("/")
def home():
    return {"message": "API is working 🚀"}


# ---------------- MATCH API ---------------- #

@router.post("/match")
def match_resume(request: MatchRequest):

    start = time.time()

    orchestrator = Orchestrator()

    result = orchestrator.process(
        request.resume_text,
        request.job_skills
    )

    end = time.time()
    result["latency"] = round(end - start, 2)

    return result


# ---------------- PARSE FILE API ---------------- #

@router.post("/parse-file")
async def parse_file(file: UploadFile = File(...)):
    start = time.time()

    content = await file.read()

    parser = ParserAgent()

    try:
        text = parser.extract_text(content, file.filename)
    except Exception as e:
        print("File extraction error:", e)
        return {
            "personal": {"name": "Unknown", "email": "", "phone": "", "location": "", "linkedin": ""},
            "skills": {"technical": [], "soft": [], "certifications": []},
            "experience": [],
            "education": [],
            "normalized_skills": [],
            "total_experience_years": 0,
            "summary": f"Error extracting text: {str(e)}"
        }

    parsed_data = parser.parse_resume(text)

    # Ensure correct structure (guard against bad LLM output)
    if not isinstance(parsed_data, dict):
        parsed_data = {}

    if "personal" not in parsed_data or not isinstance(parsed_data.get("personal"), dict):
        parsed_data["personal"] = {"name": "Unknown", "email": "", "phone": "", "location": ""}

    if "skills" not in parsed_data or not isinstance(parsed_data.get("skills"), dict):
        parsed_data["skills"] = {"technical": [], "soft": []}

    # Ensure personal has all expected keys
    personal = parsed_data["personal"]
    personal.setdefault("name", "Unknown")
    personal.setdefault("email", "")
    personal.setdefault("phone", "")
    personal.setdefault("location", "")
    personal.setdefault("linkedin", "")

    # Ensure skills has all expected keys
    skills = parsed_data["skills"]
    skills.setdefault("technical", [])
    skills.setdefault("soft", [])
    skills.setdefault("certifications", [])

    # Normalization
    try:
        normalizer = NormalizationAgent()
        norm_result = normalizer.normalize_skills(
            skills.get("technical", []),
            text
        )
        parsed_data["normalized_skills"] = norm_result.get("normalized_skills", [])
        parsed_data["skill_hierarchy"] = norm_result.get("hierarchy", {})
        parsed_data["skill_proficiency"] = norm_result.get("proficiency", {})
        parsed_data["unknown_skills"] = norm_result.get("unknown_skills", [])
    except Exception as e:
        print("Normalization error in /parse-file:", e)
        parsed_data["normalized_skills"] = skills.get("technical", [])
        parsed_data["skill_hierarchy"] = {}
        parsed_data["skill_proficiency"] = {}
        parsed_data["unknown_skills"] = []

    # Clean name (safe)
    try:
        name = personal.get("name", "Unknown") or "Unknown"
        name_parts = name.split()
        if len(name_parts) >= 2:
            personal["name"] = name_parts[0] + " " + name_parts[1]
    except Exception:
        pass

    # Ensure other top-level keys
    parsed_data.setdefault("experience", [])
    parsed_data.setdefault("education", [])
    parsed_data.setdefault("inferred_skills", [])
    parsed_data.setdefault("total_experience_years", 0)
    parsed_data.setdefault("summary", "Parsed from backend")

    end = time.time()
    parsed_data["latency"] = round(end - start, 2)

    return parsed_data


# ---------------- BATCH PARSE API ---------------- #

@router.post("/parse-batch")
async def parse_batch(files: List[UploadFile] = File(...)):
    results = []

    parser = ParserAgent()
    normalizer = NormalizationAgent()

    for file in files:
        content = await file.read()

        text = parser.extract_text(content, file.filename)
        parsed_data = parser.parse_resume(text)

        if "skills" not in parsed_data or not isinstance(parsed_data["skills"], dict):
            parsed_data["skills"] = {"technical": [], "soft": []}

        norm = normalizer.normalize_skills(parsed_data["skills"]["technical"], text)

        parsed_data["normalized_skills"] = norm.get("normalized_skills", [])

        results.append({
            "filename": file.filename,
            "data": parsed_data
        })

    return {"results": results}

@router.get("/skills/taxonomy")
def get_taxonomy():
    from app.agents.normalization_agent import SKILL_TAXONOMY
    return SKILL_TAXONOMY


@router.get("/health")
def health():
    return {"status": "running"}