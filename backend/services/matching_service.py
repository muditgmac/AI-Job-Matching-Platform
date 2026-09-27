import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for matching resumes with job descriptions using semantic similarity"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize matching service with sentence transformer model

        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        """Normalize a skill name for deterministic comparison."""
        return " ".join(skill.lower().strip().split())

    def analyze_skill_match(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Produce an explainable skill-match analysis.

        Required skills carry more weight than preferred skills when both
        groups are present. The returned structure preserves the job's
        original skill names while matching them case-insensitively.

        Returns:
            Dictionary containing required/preferred coverage scores,
            matched skills, missing skills, and a combined skill score.
        """
        preferred_skills = preferred_skills or []

        resume_normalized = {
            self._normalize_skill(skill)
            for skill in resume_skills
            if skill and skill.strip()
        }

        required_pairs = [
            (skill, self._normalize_skill(skill))
            for skill in required_skills
            if skill and skill.strip()
        ]

        preferred_pairs = [
            (skill, self._normalize_skill(skill))
            for skill in preferred_skills
            if skill and skill.strip()
        ]

        matched_required = [
            original
            for original, normalized in required_pairs
            if normalized in resume_normalized
        ]

        missing_required = [
            original
            for original, normalized in required_pairs
            if normalized not in resume_normalized
        ]

        matched_preferred = [
            original
            for original, normalized in preferred_pairs
            if normalized in resume_normalized
        ]

        missing_preferred = [
            original
            for original, normalized in preferred_pairs
            if normalized not in resume_normalized
        ]

        required_score = (
            len(matched_required) / len(required_pairs)
            if required_pairs
            else 1.0
        )

        preferred_score = (
            len(matched_preferred) / len(preferred_pairs)
            if preferred_pairs
            else 1.0
        )

        if required_pairs and preferred_pairs:
            combined_score = 0.8 * required_score + 0.2 * preferred_score
        elif required_pairs:
            combined_score = required_score
        elif preferred_pairs:
            combined_score = preferred_score
        else:
            combined_score = 1.0

        return {
            "score": float(combined_score),
            "required_score": float(required_score),
            "preferred_score": float(preferred_score),
            "matched_required_skills": matched_required,
            "missing_required_skills": missing_required,
            "matched_preferred_skills": matched_preferred,
            "missing_preferred_skills": missing_preferred,
        }

    def calculate_skill_match_score(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> float:
        """
        Calculate skill matching score while preserving the original
        public interface.
        """
        analysis = self.analyze_skill_match(
            resume_skills=resume_skills,
            required_skills=job_skills
        )
        return analysis["score"]

    def calculate_experience_score(self, resume_experience: List[Dict], job_experience_level: Optional[str] = None) -> float:
        """
        Calculate experience relevance score

        Args:
            resume_experience: List of experience entries from resume
            job_experience_level: Required experience level (entry/mid/senior)

        Returns:
            Experience score between 0 and 1
        """
        if not resume_experience:
            return 0.0

        # Count years of experience (rough estimate)
        total_years = len(resume_experience)  # Simplified: each entry = ~1-2 years

        # Score based on experience level
        if job_experience_level:
            level = job_experience_level.lower()
            if level == 'entry' and total_years >= 0:
                return min(total_years / 2, 1.0)
            elif level == 'mid' and total_years >= 2:
                return min((total_years - 2) / 3, 1.0)
            elif level == 'senior' and total_years >= 5:
                return min((total_years - 5) / 5, 1.0)

        # Default: score based on having experience
        return min(total_years / 5, 1.0)

    def calculate_education_score(self, resume_education: List[Dict], job_requirements: Optional[Dict] = None) -> float:
        """
        Calculate education alignment score

        Args:
            resume_education: List of education entries from resume
            job_requirements: Optional job education requirements

        Returns:
            Education score between 0 and 1
        """
        if not resume_education:
            return 0.5  # Neutral score if no education info

        # If education exists, give base score
        has_degree = any(edu.get('degree') for edu in resume_education)

        if has_degree:
            return 1.0

        return 0.5

    def calculate_overall_score(
        self,
        semantic_similarity: float,
        skill_match: float,
        experience_score: float,
        education_score: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted overall matching score

        Args:
            semantic_similarity: Semantic similarity score
            skill_match: Skill matching score
            experience_score: Experience score
            education_score: Education score
            weights: Optional custom weights for each factor

        Returns:
            Overall score between 0 and 1
        """
        if weights is None:
            weights = {
                'semantic': 0.4,
                'skills': 0.3,
                'experience': 0.2,
                'education': 0.1
            }

        overall = (
            semantic_similarity * weights['semantic'] +
            skill_match * weights['skills'] +
            experience_score * weights['experience'] +
            education_score * weights['education']
        )

        return float(overall)

    def build_match_explanation(
        self,
        semantic_similarity: float,
        skill_match: float,
        experience_score: float,
        education_score: float,
        skill_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a deterministic explanation for a hybrid match score.

        The explanation exposes component scores and skill-level evidence
        without requiring an additional LLM call.
        """
        skill_details = skill_details or {}

        components = {
            "semantic_similarity": float(semantic_similarity),
            "skill_match": float(skill_match),
            "experience": float(experience_score),
            "education": float(education_score),
        }

        weighted_contributions = {
            "semantic_similarity": float(semantic_similarity * 0.4),
            "skill_match": float(skill_match * 0.3),
            "experience": float(experience_score * 0.2),
            "education": float(education_score * 0.1),
        }

        # Rank explanatory importance by actual contribution to the
        # weighted overall score, not by the raw component value.
        strongest_component = max(
            weighted_contributions,
            key=weighted_contributions.get
        )

        weakest_component = min(
            weighted_contributions,
            key=weighted_contributions.get
        )

        matched_required = skill_details.get(
            "matched_required_skills", []
        )
        missing_required = skill_details.get(
            "missing_required_skills", []
        )
        matched_preferred = skill_details.get(
            "matched_preferred_skills", []
        )
        missing_preferred = skill_details.get(
            "missing_preferred_skills", []
        )

        return {
            "components": components,
            "weighted_contributions": weighted_contributions,
            "strongest_component": strongest_component,
            "weakest_component": weakest_component,
            "matched_required_skills": matched_required,
            "missing_required_skills": missing_required,
            "matched_preferred_skills": matched_preferred,
            "missing_preferred_skills": missing_preferred,
        }

    def match_resume_to_job(
        self,
        resume_text: str,
        resume_skills: List[str],
        resume_experience: List[Dict],
        resume_education: List[Dict],
        job_description: str,
        job_skills: List[str],
        job_experience_level: Optional[str] = None,
        job_preferred_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Match a resume to a job description and return scores

        Args:
            resume_text: Full text of the resume
            resume_skills: Skills extracted from resume
            resume_experience: Experience entries from resume
            resume_education: Education entries from resume
            job_description: Job description text
            job_skills: Required/preferred skills for the job
            job_experience_level: Required experience level

        Returns:
            Dictionary with all matching scores
        """
        # Generate embeddings
        resume_embedding = self.generate_embedding(resume_text)
        job_embedding = self.generate_embedding(job_description)

        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(resume_embedding, job_embedding)

        # Analyze required and preferred skills separately so the main
        # matching path uses the same explainable skill logic.
        skill_details = self.analyze_skill_match(
            resume_skills=resume_skills,
            required_skills=job_skills,
            preferred_skills=job_preferred_skills,
        )
        skill_match = skill_details["score"]

        # Calculate experience score
        experience_score = self.calculate_experience_score(resume_experience, job_experience_level)

        # Calculate education score
        education_score = self.calculate_education_score(resume_education)

        # Calculate overall score
        overall_score = self.calculate_overall_score(
            semantic_sim, skill_match, experience_score, education_score
        )

        explanation = self.build_match_explanation(
            semantic_similarity=semantic_sim,
            skill_match=skill_match,
            experience_score=experience_score,
            education_score=education_score,
            skill_details=skill_details,
        )

        return {
            'overall_score': overall_score,
            'semantic_similarity': semantic_sim,
            'skill_match_score': skill_match,
            'experience_score': experience_score,
            'education_score': education_score,
            'skill_details': skill_details,
            'explanation': explanation,
            'resume_embedding': resume_embedding,
            'job_embedding': job_embedding
        }
