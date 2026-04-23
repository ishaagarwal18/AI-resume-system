class Orchestrator:

    def __init__(self):
        from app.agents.parser_agent import ParserAgent
        from app.agents.normalization_agent import NormalizationAgent
        from app.agents.matcher_agent import MatcherAgent

        self.parser = ParserAgent()
        self.normalizer = NormalizationAgent()
        self.matcher = MatcherAgent()

    def process(self, text, job_description):
        try:
            parsed = self.parser.parse_resume(text)

            if "skills" not in parsed or not isinstance(parsed["skills"], dict):
                parsed["skills"] = {"technical": [], "soft": []}

            normalized = self.normalizer.normalize_skills(
                parsed["skills"]["technical"], text
            )

            match = self.matcher.match(
                normalized.get("normalized_skills", []),
                job_description
            )

            match["evaluation"] = {
                "matched_count": len(match.get("matched_skills", [])),
                "missing_count": len(match.get("missing_skills", [])),
                "confidence": "high" if match.get("overall_score", 0) > 70 else "medium"
            }

            return {
                "parsed": parsed,
                "normalized": normalized,
                "match": match
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "partial_failure"
            }
