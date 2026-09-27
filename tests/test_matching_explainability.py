import unittest

from backend.services.matching_service import MatchingService


class MatchingExplainabilityTests(unittest.TestCase):

    def setUp(self):
        self.service = MatchingService.__new__(MatchingService)

    def test_skill_matching_is_case_insensitive(self):
        result = self.service.analyze_skill_match(
            resume_skills=["Python", "FASTAPI", "PostgreSQL"],
            required_skills=["python", "FastAPI", "Docker"],
        )

        self.assertAlmostEqual(result["required_score"], 2 / 3)
        self.assertEqual(
            result["matched_required_skills"],
            ["python", "FastAPI"],
        )
        self.assertEqual(
            result["missing_required_skills"],
            ["Docker"],
        )

    def test_required_skills_are_weighted_more_than_preferred(self):
        result = self.service.analyze_skill_match(
            resume_skills=["Python", "SQL", "Docker"],
            required_skills=["Python", "SQL"],
            preferred_skills=["Docker", "Kubernetes"],
        )

        self.assertEqual(result["required_score"], 1.0)
        self.assertEqual(result["preferred_score"], 0.5)
        self.assertAlmostEqual(result["score"], 0.9)

    def test_no_job_skills_produces_full_skill_score(self):
        result = self.service.analyze_skill_match(
            resume_skills=["Python"],
            required_skills=[],
            preferred_skills=[],
        )

        self.assertEqual(result["score"], 1.0)

    def test_existing_skill_score_interface_is_preserved(self):
        score = self.service.calculate_skill_match_score(
            resume_skills=["Python", "SQL"],
            job_skills=["python", "Docker"],
        )

        self.assertEqual(score, 0.5)

    def test_explanation_contains_weighted_contributions(self):
        skill_details = self.service.analyze_skill_match(
            resume_skills=["Python", "FastAPI"],
            required_skills=["Python", "FastAPI", "Docker"],
        )

        explanation = self.service.build_match_explanation(
            semantic_similarity=0.8,
            skill_match=skill_details["score"],
            experience_score=0.6,
            education_score=1.0,
            skill_details=skill_details,
        )

        self.assertAlmostEqual(
            explanation["weighted_contributions"]["semantic_similarity"],
            0.32,
        )

        self.assertEqual(
            explanation["missing_required_skills"],
            ["Docker"],
        )

        self.assertEqual(
            explanation["strongest_component"],
            "semantic_similarity",
        )

    def test_main_match_path_uses_preferred_skills_and_explanation(self):
        # Avoid loading the transformer model: this test isolates orchestration.
        self.service.generate_embedding = lambda text: [1.0, 0.0]
        self.service.calculate_semantic_similarity = lambda a, b: 0.8
        self.service.calculate_experience_score = lambda experience, level: 0.6
        self.service.calculate_education_score = lambda education: 1.0

        result = self.service.match_resume_to_job(
            resume_text="Python FastAPI developer",
            resume_skills=["Python", "FastAPI"],
            resume_experience=[{"title": "Engineer"}],
            resume_education=[{"degree": "BS"}],
            job_description="Backend engineering role",
            job_skills=["Python", "Docker"],
            job_experience_level="entry",
            job_preferred_skills=["FastAPI", "Kubernetes"],
        )

        # required = 1/2, preferred = 1/2
        # combined skill score = 0.8(0.5) + 0.2(0.5) = 0.5
        self.assertAlmostEqual(result["skill_match_score"], 0.5)

        self.assertEqual(
            result["skill_details"]["matched_preferred_skills"],
            ["FastAPI"],
        )
        self.assertEqual(
            result["skill_details"]["missing_required_skills"],
            ["Docker"],
        )
        self.assertEqual(
            result["explanation"]["strongest_component"],
            "semantic_similarity",
        )


if __name__ == "__main__":
    unittest.main()
