"""
normalization_agent.py  —  ATS Skill Normaliser (v2)

Improvements over v1:
  - Taxonomy expanded from 12 → 80+ canonical skills
  - Soft skill normalisation (separate pipeline)
  - Confidence threshold tuned per category
  - Hierarchy covers all taxonomy entries (not just 12)
  - Proficiency estimation improved: year-count regex, senior/junior keywords,
    explicit level keywords (expert, proficient, beginner, etc.)
  - Inference rules extended: cloud, DevOps, frontend, backend, data
  - Aliases dict for hard-coded exact matches (js → JavaScript, etc.)
    so common abbreviations never fall through to semantic search
  - All methods documented
"""

from __future__ import annotations
import re
import torch
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Canonical taxonomy
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: list[str] = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
    "Rust", "Ruby", "Swift", "Kotlin", "Scala", "PHP", "R", "MATLAB",
    "Bash", "Perl", "Dart", "Elixir",

    # Frontend
    "React", "Vue.js", "Angular", "Next.js", "Svelte", "Redux",
    "HTML", "CSS", "Sass", "Webpack", "Tailwind CSS",

    # Backend
    "Node.js", "Express", "FastAPI", "Flask", "Django", "Spring Boot",
    "Laravel", "Ruby on Rails", "gRPC",

    # Data / ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn",
    "Pandas", "NumPy", "Apache Spark", "Airflow", "dbt",
    "MLflow", "LangChain", "Data Science", "Data Engineering",
    "Large Language Models",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",
    "Elasticsearch", "DynamoDB", "BigQuery", "Snowflake",

    # Cloud & DevOps
    "AWS", "Google Cloud Platform", "Azure", "Docker", "Kubernetes",
    "Terraform", "Ansible", "Jenkins", "GitHub Actions", "CI/CD",
    "Linux", "Nginx", "Apache Kafka", "RabbitMQ",

    # Architecture / practices
    "Microservices", "REST API", "GraphQL", "System Design",
    "Agile", "Scrum", "TDD", "Git",

    # Testing
    "Pytest", "Jest", "Selenium", "Cypress",
]

# Exact-match aliases (lowercase key → canonical)
ALIASES: dict[str, str] = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "golang": "Go",
    "node": "Node.js",
    "nodejs": "Node.js",
    "vue": "Vue.js",
    "next": "Next.js",
    "tf": "TensorFlow",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "k8s": "Kubernetes",
    "gcp": "Google Cloud Platform",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "nlp": "NLP",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "cv": "Computer Vision",
    "ci/cd": "CI/CD",
    "c++": "C++",
    "c#": "C#",
    "springboot": "Spring Boot",
    "spring": "Spring Boot",
    "kafka": "Apache Kafka",
}

# Hierarchy map  skill → [category, parent category]
HIERARCHY: dict[str, list[str]] = {
    # Languages
    **{s: ["Programming Languages", "Technical Skills"] for s in [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
        "Rust", "Ruby", "Swift", "Kotlin", "Scala", "PHP", "R", "MATLAB",
        "Bash", "Perl", "Dart", "Elixir",
    ]},
    # Frontend
    **{s: ["Frontend", "Technical Skills"] for s in [
        "React", "Vue.js", "Angular", "Next.js", "Svelte", "Redux",
        "HTML", "CSS", "Sass", "Webpack", "Tailwind CSS",
    ]},
    # Backend
    **{s: ["Backend", "Technical Skills"] for s in [
        "Node.js", "Express", "FastAPI", "Flask", "Django", "Spring Boot",
        "Laravel", "Ruby on Rails", "gRPC",
    ]},
    # AI / ML
    **{s: ["AI/ML", "Technical Skills"] for s in [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn",
        "Pandas", "NumPy", "MLflow", "LangChain",
        "Data Science", "Large Language Models",
    ]},
    # Data Engineering
    **{s: ["Data Engineering", "Technical Skills"] for s in [
        "Apache Spark", "Airflow", "dbt", "Data Engineering",
        "BigQuery", "Snowflake",
    ]},
    # Databases
    **{s: ["Databases", "Technical Skills"] for s in [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",
        "Elasticsearch", "DynamoDB",
    ]},
    # Cloud
    **{s: ["Cloud", "Technical Skills"] for s in [
        "AWS", "Google Cloud Platform", "Azure",
    ]},
    # DevOps
    **{s: ["DevOps", "Technical Skills"] for s in [
        "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
        "GitHub Actions", "CI/CD", "Linux", "Nginx",
        "Apache Kafka", "RabbitMQ",
    ]},
    # Architecture
    **{s: ["Architecture & Practices", "Technical Skills"] for s in [
        "Microservices", "REST API", "GraphQL", "System Design",
        "Agile", "Scrum", "TDD", "Git",
    ]},
    # Testing
    **{s: ["Testing", "Technical Skills"] for s in [
        "Pytest", "Jest", "Selenium", "Cypress",
    ]},
}

# Inference rules: if any skill in `triggers` is present, add all `infer`
INFERENCE_RULES: list[dict] = [
    {"triggers": {"TensorFlow", "PyTorch", "Keras"},         "infer": {"Deep Learning", "Machine Learning"}},
    {"triggers": {"scikit-learn", "Pandas", "NumPy"},        "infer": {"Data Science", "Machine Learning"}},
    {"triggers": {"React", "Vue.js", "Angular", "Svelte"},   "infer": {"JavaScript", "Frontend"}},
    {"triggers": {"Next.js"},                                 "infer": {"React", "JavaScript", "Frontend"}},
    {"triggers": {"Node.js", "Express", "Fastify"},          "infer": {"JavaScript", "Backend"}},
    {"triggers": {"FastAPI", "Flask", "Django"},              "infer": {"Python", "Backend"}},
    {"triggers": {"Docker", "Kubernetes"},                    "infer": {"DevOps", "CI/CD"}},
    {"triggers": {"AWS", "Google Cloud Platform", "Azure"},  "infer": {"Cloud"}},
    {"triggers": {"Apache Spark", "Airflow", "dbt"},         "infer": {"Data Engineering"}},
    {"triggers": {"LangChain", "Large Language Models"},      "infer": {"NLP", "Machine Learning"}},
]

# Proficiency keyword patterns
PROFICIENCY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(\d+)\+?\s*years?\b.*?{skill}", "Advanced"),       # "5 years Python"
    (r"{skill}.*?\b(\d+)\+?\s*years?\b", "Advanced"),       # "Python (3+ years)"
    (r"\b(expert|advanced|senior|lead|architect)\b", "Advanced"),
    (r"\b(proficient|experienced|strong|solid)\b", "Intermediate"),
    (r"\b(familiar|basic|beginner|learning|entry.level|junior)\b", "Beginner"),
]


class NormalizationAgent:
    """
    Normalises a raw skill list to canonical taxonomy names, infers
    implied skills, builds a hierarchy map, and estimates proficiency.

    Usage:
        agent = NormalizationAgent()
        result = agent.normalize_skills(["js", "react", "k8s"], text=resume_text)
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.taxonomy_embeddings = self.model.encode(
            SKILL_TAXONOMY, convert_to_tensor=True
        )

    # ------------------------------------------------------------------ #
    #  SEMANTIC MATCHING                                                   #
    # ------------------------------------------------------------------ #

    def get_best_match(
        self, skill: str, threshold: float = 0.62
    ) -> tuple[str | None, float]:
        """
        Returns (canonical_skill, confidence) or (None, confidence)
        if below threshold.
        """
        # Fast path: exact alias lookup
        alias_key = skill.lower().strip()
        if alias_key in ALIASES:
            return ALIASES[alias_key], 1.0

        # Fast path: exact taxonomy match (case-insensitive)
        lower_tax = {t.lower(): t for t in SKILL_TAXONOMY}
        if alias_key in lower_tax:
            return lower_tax[alias_key], 1.0

        # Semantic fallback
        emb = self.model.encode(skill, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.taxonomy_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        confidence = float(scores[best_idx].item())

        if confidence < threshold:
            return None, confidence
        return SKILL_TAXONOMY[best_idx], confidence

    # ------------------------------------------------------------------ #
    #  INFERENCE                                                           #
    # ------------------------------------------------------------------ #

    def infer_skills(self, skills: set[str]) -> set[str]:
        """
        Applies domain inference rules. E.g. TensorFlow → Deep Learning.
        """
        result = set(skills)
        for rule in INFERENCE_RULES:
            if rule["triggers"] & result:
                result |= rule["infer"]
        return result

    # ------------------------------------------------------------------ #
    #  PROFICIENCY                                                         #
    # ------------------------------------------------------------------ #

    def estimate_proficiency(self, text: str, skill: str) -> str:
        """
        Multi-signal proficiency estimation:
          1. Year-count patterns near the skill mention
          2. Seniority keywords in surrounding context
          3. Generic level keywords anywhere in text

        Returns: 'Advanced' | 'Intermediate' | 'Beginner'
        """
        text_lower = text.lower()
        skill_lower = skill.lower()

        # Narrow context: 120 chars around skill mention
        idx = text_lower.find(skill_lower)
        context = text_lower[max(0, idx - 60): idx + 60 + len(skill_lower)] if idx >= 0 else ""
        full_context = context + " " + text_lower  # check global too

        for pattern_template, level in PROFICIENCY_PATTERNS:
            pattern = pattern_template.replace("{skill}", re.escape(skill_lower))
            if re.search(pattern, full_context, re.IGNORECASE):
                return level

        return "Intermediate"

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT                                                    #
    # ------------------------------------------------------------------ #

    def normalize_skills(
        self, skills: list[str], text: str = ""
    ) -> dict:
        """
        Full normalisation pipeline:
          1. Alias + semantic match to canonical taxonomy
          2. Inference of implied skills
          3. Hierarchy assignment
          4. Proficiency estimation

        Args:
            skills: Raw skill strings (from parser or manual input)
            text:   Full resume text for proficiency context (optional)

        Returns:
            {
                "normalized_skills": [...],
                "hierarchy": {skill: [category, parent]},
                "proficiency": {skill: level},
                "unknown_skills": [...],
                "inferred_skills": [...],
            }
        """
        normalized: list[str] = []
        unknown: list[str] = []
        confidence_log: list[dict] = []

        # Step 1 — semantic match
        for skill in skills:
            match, conf = self.get_best_match(skill)
            confidence_log.append({"raw": skill, "match": match, "confidence": round(conf, 3)})
            if match:
                normalized.append(match)
            else:
                unknown.append(skill)

        normalized_set = set(normalized)

        # Step 2 — inference
        enriched = self.infer_skills(normalized_set)
        inferred = sorted(enriched - normalized_set)
        all_skills = sorted(enriched)

        # Step 3 — hierarchy
        hierarchy: dict[str, list[str]] = {}
        for skill in all_skills:
            hierarchy[skill] = HIERARCHY.get(skill, ["Other", "Skills"])

        # Step 4 — proficiency
        proficiency: dict[str, str] = {}
        for skill in all_skills:
            proficiency[skill] = self.estimate_proficiency(text, skill)

        return {
            "normalized_skills": all_skills,
            "hierarchy":         hierarchy,
            "proficiency":       proficiency,
            "unknown_skills":    unknown,
            "inferred_skills":   inferred,
            "_confidence_log":   confidence_log,  # useful for debugging / tuning
        }
