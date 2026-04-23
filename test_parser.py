from app.agents.parser_agent import ParserAgent
from app.agents.normalization_agent import NormalizationAgent
from app.agents.matcher_agent import MatcherAgent

# Sample resume
sample_resume = """
Vyom Prajapati
vyom@gmail.com
+91 9876543210

Skills:
Python, React.js, JS, ML
"""

# Sample job description skills
job_skills = ["Python", "Docker", "React"]

# Step 1: Parse
parser = ParserAgent()
parsed_data = parser.parse_resume(sample_resume)

# Step 2: Normalize
normalizer = NormalizationAgent()
normalized_skills = normalizer.normalize_skills(parsed_data["skills"])

# Step 3: Match
matcher = MatcherAgent()
result = matcher.match(normalized_skills, job_skills)

print("Normalized Skills:", normalized_skills)
print("Match Result:", result)