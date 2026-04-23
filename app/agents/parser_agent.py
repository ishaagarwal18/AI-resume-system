"""
parser_agent.py  —  ATS Resume Parser (v2)

Improvements over v1:
  - Expanded rule-based skills DB (100+ skills)
  - Robust name extraction with heuristics + fallback
  - Temp files written to /tmp to avoid permission issues
  - LLM fallback triggered when structural sections are weak, not just skills count
  - JSON extraction hardened: strips ```json ... ``` fences properly
  - Section splitting (experience, education, projects, certifications)
  - Phone normalisation + location heuristic
  - All public methods documented
"""

import re
import json
import tempfile
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Skill taxonomy (expandable — keep lowercase for matching)
# ---------------------------------------------------------------------------
SKILLS_DB = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
    "rust", "ruby", "swift", "kotlin", "scala", "php", "r", "matlab", "perl",
    "bash", "shell", "powershell", "dart", "elixir", "clojure", "haskell",

    # Frontend
    "react", "vue", "angular", "next.js", "nuxt", "svelte", "redux", "tailwind",
    "html", "css", "sass", "webpack", "vite", "graphql", "rest api",

    # Backend / infra
    "node", "node.js", "express", "fastapi", "flask", "django", "spring", "laravel",
    "rails", "gin", "fiber", "grpc",

    # Data / ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "spark",
    "hadoop", "airflow", "dbt", "mlflow", "hugging face", "langchain", "openai",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "data science", "data engineering",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "bigquery", "snowflake", "oracle",

    # DevOps / cloud
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions",
    "circleci", "aws", "gcp", "azure", "linux", "nginx", "kafka", "rabbitmq",

    # Testing
    "pytest", "jest", "selenium", "cypress", "unit testing", "tdd",

    # Tools / practices
    "git", "jira", "agile", "scrum", "ci/cd", "microservices", "system design",
    "api design",
}

# Soft skills (separate so they don't pollute technical matching)
SOFT_SKILLS_DB = {
    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
    "time management", "adaptability", "collaboration", "mentoring", "project management",
    "stakeholder management", "presentation",
}

# Section header keywords
SECTION_HEADERS = {
    "experience":     ["experience", "work history", "employment", "professional background"],
    "education":      ["education", "academic", "qualification", "degree"],
    "skills":         ["skills", "technical skills", "competencies", "technologies"],
    "projects":       ["projects", "side projects", "portfolio", "open source"],
    "certifications": ["certifications", "certificates", "credentials", "licenses"],
    "summary":        ["summary", "profile", "objective", "about me"],
}


class ParserAgent:
    """
    Parses resumes from .txt, .docx, or .pdf bytes.

    Usage:
        agent = ParserAgent()
        text  = agent.extract_text(file_bytes, "resume.pdf")
        data  = agent.parse_resume(text)
    """

    # ------------------------------------------------------------------ #
    #  FILE EXTRACTION                                                     #
    # ------------------------------------------------------------------ #

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Convert raw file bytes to a plain-text string.
        Supports .txt, .docx, .pdf.
        """
        filename = filename.lower()

        if filename.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")

        elif filename.endswith(".docx"):
            return self._extract_docx(file_bytes)

        elif filename.endswith(".pdf"):
            return self._extract_pdf(file_bytes)

        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def _extract_docx(self, file_bytes: bytes) -> str:
        from docx import Document
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name
        try:
            doc = Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs)
        finally:
            os.unlink(tmp_path)

    def _extract_pdf(self, file_bytes: bytes) -> str:
        import pdfplumber
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name
        try:
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            return text
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    #  RULE-BASED EXTRACTORS                                               #
    # ------------------------------------------------------------------ #

    def extract_email(self, text: str) -> str:
        match = re.search(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", text)
        return match.group(0).lower() if match else ""

    def extract_phone(self, text: str) -> str:
        """
        Matches international and local phone formats.
        Returns the first match, stripped of surrounding whitespace.
        """
        pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
        match = re.search(pattern, text)
        if match:
            raw = match.group(0).strip()
            # Remove ambiguous trailing characters
            return re.sub(r"[\s\-().]+$", "", raw)
        return ""

    def extract_name(self, text: str) -> str:
        """
        Heuristic name extraction:
        1. Skip lines that look like emails / section headers / dates / URLs.
        2. Accept lines of 2-4 capitalised tokens within the first 8 lines.
        3. Fall back to 'Unknown'.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        skip_patterns = re.compile(
            r"(@|http|www\.|skills|experience|education|summary|resume|curriculum|"
            r"\d{4}|\+\d|\bphone\b|\bemail\b|linkedin|github)",
            re.IGNORECASE,
        )

        for line in lines[:8]:
            if skip_patterns.search(line):
                continue
            tokens = line.split()
            if 2 <= len(tokens) <= 4 and all(t[0].isupper() for t in tokens if t.isalpha()):
                return line
        return "Unknown"

    def extract_location(self, text: str) -> str:
        """
        Light heuristic: look for 'City, State/Country' pattern near the top.
        """
        pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*([A-Z]{2,}|[A-Z][a-z]+)\b"
        )
        for line in text.split("\n")[:15]:
            m = pattern.search(line)
            if m and "@" not in line:
                return m.group(0)
        return ""

    def extract_skills(self, text: str) -> dict:
        """
        Returns {'technical': [...], 'soft': [...]}.
        Skills section text is weighted higher by scanning it first,
        then the full text is scanned for anything missed.
        """
        text_lower = text.lower()
        technical = set()
        soft = set()

        for skill in SKILLS_DB:
            # Use word-boundary aware search
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                technical.add(skill.title())

        for skill in SOFT_SKILLS_DB:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                soft.add(skill.title())

        return {
            "technical": sorted(technical),
            "soft": sorted(soft),
        }

    def extract_sections(self, text: str) -> dict:
        """
        Splits the resume into labelled sections using header detection.
        Returns a dict of {section_name: section_text}.
        """
        lines = text.split("\n")
        sections = {}
        current_section = "header"
        buffer = []

        header_map = {}
        for canon, aliases in SECTION_HEADERS.items():
            for alias in aliases:
                header_map[alias.lower()] = canon

        for line in lines:
            stripped = line.strip().lower().rstrip(":").rstrip("s")  # plural tolerance
            # Check if line is a section header (short, no punctuation overload)
            if stripped in header_map and len(stripped) < 30:
                if buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = header_map[stripped]
                buffer = []
            else:
                buffer.append(line)

        if buffer:
            sections[current_section] = "\n".join(buffer).strip()

        return sections

    def parse_experience(self, text: str) -> list:
        """
        Simple bullet/role extraction from experience section text.
        """
        entries = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 20:  # skip tiny fragments
                entries.append(line)
        return entries[:20]  # cap for safety

    # ------------------------------------------------------------------ #
    #  LLM EXTRACTION                                                      #
    # ------------------------------------------------------------------ #

    def _clean_json_response(self, content: str) -> str:
        """Strip markdown code fences from LLM output."""
        content = content.strip()
        # Handle ```json ... ``` and ``` ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
        if fence:
            return fence.group(1).strip()
        return content

    def extract_with_llm(self, text: str) -> dict | None:
        """
        Uses a local Ollama llama3 model for deep extraction.
        Returns parsed dict or None on failure.
        """
        try:
            import ollama
        except ImportError:
            print("ollama not installed — skipping LLM extraction.")
            return None

        prompt = f"""You are a strict resume parser.
Extract structured data and return ONLY valid JSON. No explanation. No text outside JSON.

Format:
{{
  "personal": {{
    "name": "",
    "email": "",
    "phone": "",
    "location": ""
  }},
  "skills": {{
    "technical": [],
    "soft": []
  }},
  "experience": [],
  "education": [],
  "projects": [],
  "certifications": []
}}

Rules:
- Always return valid JSON
- If data missing, use "" or []
- Do NOT add any text outside the JSON object

Resume:
{text[:3000]}
"""

        try:
            response = ollama.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"]
            content = self._clean_json_response(content)
            return json.loads(content)
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  MAIN PARSE FUNCTION                                                 #
    # ------------------------------------------------------------------ #

    def parse_resume(self, text: str) -> dict:
        """
        Full pipeline:
          1. Rule-based extraction (fast, always runs)
          2. Section splitting for experience / education / etc.
          3. LLM deep extraction if rule-based result looks thin

        Returns a structured dict ready for downstream agents.
        """
        # Step 1 — rule-based
        skills = self.extract_skills(text)
        sections = self.extract_sections(text)

        basic_data = {
            "personal": {
                "name":     self.extract_name(text),
                "email":    self.extract_email(text),
                "phone":    self.extract_phone(text),
                "location": self.extract_location(text),
            },
            "skills": skills,
            "experience":     self.parse_experience(sections.get("experience", "")),
            "education":      [l.strip() for l in sections.get("education", "").split("\n") if l.strip()],
            "projects":       [l.strip() for l in sections.get("projects", "").split("\n") if l.strip()],
            "certifications": [l.strip() for l in sections.get("certifications", "").split("\n") if l.strip()],
            "_extraction_method": "rule_based",
        }

        # Step 2 — decide if LLM is needed
        tech_count = len(skills["technical"])
        has_experience = bool(basic_data["experience"])
        personal_complete = all([
            basic_data["personal"]["name"] != "Unknown",
            basic_data["personal"]["email"],
        ])

        needs_llm = tech_count < 3 or not has_experience or not personal_complete

        if needs_llm:
            print("Rule-based result is thin — invoking LLM for deeper extraction...")
            llm_result = self.extract_with_llm(text)
            if llm_result:
                # Merge: LLM wins on personal + experience + education
                # but we keep rule-based skills if they're richer
                llm_result["_extraction_method"] = "llm"
                llm_tech = llm_result.get("skills", {}).get("technical", [])
                if tech_count > len(llm_tech):
                    llm_result["skills"]["technical"] = skills["technical"]
                return llm_result

        return basic_data
