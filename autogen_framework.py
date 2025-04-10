"""
AutoGen-based Multi-Agent Framework for Talent Matching Tool

This module implements a two-agent framework using Microsoft's AutoGen library:
1. Coordinator Agent: Analyzes CVs and filters candidates based on initial criteria
2. HR Manager Agent: Performs in-depth analysis and ranks suitable candidates

Both agents use AutoGen for communication and reasoning.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import autogen
from pydantic import ValidationError

# Import the Pydantic models
from pydantic_models import JobRequirement, CandidateProfile, Skill, MatchResult

# Import Gemini integration if available
try:
    from gemini_integrations import get_gemini_config_from_env
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("talent_matching_autogen")

class AutoGenTalentMatcher:
    """
    Multi-Agent Framework using Microsoft's AutoGen for talent matching.
    """
    
    def __init__(self, config_list=None, verbose=True, use_gemini=False):
        """
        Initialize the AutoGen-based Talent Matcher.
        
        Args:
            config_list: Configuration for the LLM (if None, agents will use function calling only)
            verbose: Whether to display detailed agent conversations
            use_gemini: Whether to use Gemini API (requires GEMINI_API_KEY in environment)
        """
        self.verbose = verbose
        self.config_list = config_list
        self.use_gemini = use_gemini
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # If use_gemini is set, try to get Gemini configuration
        if use_gemini and GEMINI_AVAILABLE:
            self.gemini_config = get_gemini_config_from_env()
            if self.gemini_config:
                self.config_list = self.gemini_config.get_gemini_for_autogen()
                logger.info("Using Gemini API for AutoGen")
            else:
                logger.warning("Gemini API requested but not configured properly. Falling back to default.")
        
        # Initialize agents
        self._setup_agents()
        
        logger.info("AutoGen Talent Matcher initialized")
    
    def _setup_agents(self):
        """Set up the AutoGen agents."""
        # Create a termination message function
        def is_termination_msg(content):
            # Check for dictionary format
            if isinstance(content, dict) and "final_candidates" in content:
                return True
            # Check for string format that might signal completion
            if isinstance(content, str) and ("ranked candidates" in content.lower() or 
                                            "matching complete" in content.lower() or
                                            "top candidates" in content.lower()):
                return True
            return False
        
        # We'll create function definitions for our tools
        function_definitions = [
            {
                'name': 'filter_candidates',
                'description': 'Filter candidates based on initial criteria',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'job': {
                            'type': 'object',
                            'description': 'Job requirement details'
                        },
                        'candidates': {
                            'type': 'array',
                            'description': 'List of candidate profiles'
                        },
                        'min_match_threshold': {
                            'type': 'number',
                            'description': 'Minimum match threshold (0.0 to 1.0)'
                        }
                    },
                    'required': ['job', 'candidates', 'min_match_threshold']
                }
            },
            {
                'name': 'rank_candidates',
                'description': 'Rank candidates based on comprehensive evaluation',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'job': {
                            'type': 'object',
                            'description': 'Job requirement details'
                        },
                        'filtered_candidates': {
                            'type': 'array',
                            'description': 'List of filtered candidate profiles'
                        },
                        'top_n': {
                            'type': 'integer',
                            'description': 'Maximum number of candidates to return'
                        }
                    },
                    'required': ['job', 'filtered_candidates', 'top_n']
                }
            }
        ]
        
        # Set up LLM config - for function-only mode
        llm_config = None  # No LLM by default
        
        # Use provided LLM config if available
        if self.config_list:
            llm_config = self.config_list
        
        # Create the coordinator agent
        # For function-only agents, we'll skip creating agents entirely
        self.coordinator_agent = None
        self.hr_manager_agent = None
        
        # Create a simplified direct approach
        logger.info("Using direct matching approach (no agents)")
        
        # We'll still create the user proxy for consistency in the interface
        self.user_proxy = autogen.UserProxyAgent(
            name="TalentMatchingSystem",
            human_input_mode="NEVER",
            description="Talent Matching System that executes functions",
            code_execution_config={"use_docker": False}  # Disable Docker requirement
        )
        
        # Register functions for the user proxy
        self._register_functions()
        

    
    def _register_functions(self):
        """Register functions that agents can call."""
        # Register our key functions
        function_map = {
            "filter_candidates": self._filter_candidates,
            "rank_candidates": self._rank_candidates_with_detail,
            # Helper functions
            "extract_candidate_info": self._extract_candidate_info,
            "evaluate_skills_match": self._evaluate_skills_match,
            "evaluate_experience_match": self._evaluate_experience_match,
            "evaluate_cv_relevance": self._evaluate_cv_relevance,
            "evaluate_location_match": self._evaluate_location_match,
            "generate_match_explanation": self._generate_match_explanation
        }
        
        self.user_proxy.register_function(function_map=function_map)
    
    def _extract_candidate_info(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from a candidate profile for easier matching.
        
        Args:
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            Dictionary with extracted key information
        """
        # This would convert from a dict back to a CandidateProfile for processing
        # but for simplicity, we'll work with the dict directly
        return {
            "name": candidate.get("name", ""),
            "skills": [skill.get("name", "").lower() for skill in candidate.get("skills", [])],
            "years_experience": candidate.get("years_of_experience", 0),
            "current_title": candidate.get("current_title", ""),
            "current_location": candidate.get("current_location", ""),
            "remote_preference": candidate.get("remote_preference", False),
            "willing_to_relocate": candidate.get("willing_to_relocate", False),
            "cv_content_length": len(candidate.get("cv_content", "") or ""),
            "has_cv": bool(candidate.get("cv_content", ""))
        }
    
    def _evaluate_skills_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        Evaluate how well a candidate's skills match the job requirements.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            A score between 0.0 and 1.0 representing skills match
        """
        # Get all skill names (lowercase for case-insensitive comparison)
        required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
        preferred_skills = {skill.get("name", "").lower() for skill in job.get("preferred_skills", [])}
        candidate_skills = {skill.get("name", "").lower() for skill in candidate.get("skills", [])}
        
        # Count matches
        required_matched = required_skills.intersection(candidate_skills)
        preferred_matched = preferred_skills.intersection(candidate_skills)
        
        # Calculate scores
        required_score = len(required_matched) / len(required_skills) if required_skills else 1.0
        preferred_score = len(preferred_matched) / len(preferred_skills) if preferred_skills else 1.0
        
        # Weight required skills more heavily
        return (required_score * 0.7) + (preferred_score * 0.3)
    
    def _evaluate_experience_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        Evaluate how well a candidate's experience matches the job requirements.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            A score between 0.0 and 1.0 representing experience match
        """
        # Check years of experience
        min_years = job.get("min_years_experience", 0) or 0
        candidate_years = candidate.get("years_of_experience", 0) or 0
        
        # Simple scoring based on years
        if candidate_years >= min_years:
            years_score = 1.0
        elif min_years > 0:
            # Partial credit for close experience
            years_score = candidate_years / min_years
        else:
            years_score = 1.0
        
        # Check experience level if specified
        level_score = 1.0
        if job.get("experience_level") and candidate.get("current_title"):
            # This is a simplified check - in a real system, you'd want
            # more sophisticated logic to compare job titles and levels
            if job.get("experience_level", "").lower() in candidate.get("current_title", "").lower():
                level_score = 1.0
            else:
                level_score = 0.5  # Partial credit
        
        return (years_score * 0.7) + (level_score * 0.3)
    
    def _evaluate_cv_relevance(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        Evaluate the relevance of a candidate's CV to the job requirements.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            A score between 0.0 and 1.0 representing CV relevance
        """
        # If no CV content, return low score
        if not candidate.get("cv_content"):
            return 0.1
        
        # Create a description combining job title, skills, and qualifications
        job_description = f"{job.get('title', '')} {job.get('description', '')}"
        
        # Add required skills to the description
        for skill in job.get("required_skills", []):
            job_description += f" {skill.get('name', '')}"
        
        # Add qualifications to the description
        for qual in job.get("qualifications", []):
            job_description += f" {qual}"
        
        # Calculate TF-IDF similarity between job description and CV
        try:
            # Fit and transform the texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([job_description, candidate.get("cv_content", "")])
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            # Return the similarity score
            return float(cosine_sim[0][0])
        except:
            # Fallback if TF-IDF fails
            # Do a simple keyword matching
            keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
            cv_words = set(re.findall(r'\b\w+\b', candidate.get("cv_content", "").lower()))
            
            common_words = keywords.intersection(cv_words)
            return len(common_words) / len(keywords) if keywords else 0.0
    
    def _evaluate_location_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        Evaluate if a candidate's location matches the job location.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            A score between 0.0 and 1.0 representing location match
        """
        # If job is remote-friendly and candidate prefers remote
        if job.get("remote_friendly") and candidate.get("remote_preference"):
            return 1.0
            
        # If job location is not specified or candidate is willing to relocate
        if not job.get("location") or candidate.get("willing_to_relocate"):
            return 1.0
            
        # If both locations are specified, check if they match
        if job.get("location") and candidate.get("current_location"):
            # Simple string matching - could be enhanced with geocoding
            if job.get("location", "").lower() in candidate.get("current_location", "").lower() or \
               candidate.get("current_location", "").lower() in job.get("location", "").lower():
                return 1.0
            else:
                return 0.3  # Partial score for location mismatch
                
        # Default moderate score if we can't determine
        return 0.5
    
    def _filter_candidates(
        self,
        job: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        min_match_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates based on initial criteria.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidates: List of dictionaries for candidate profiles
            min_match_threshold: Minimum match threshold (0.0 to 1.0)
            
        Returns:
            List of dictionaries with filtered candidates and match data
        """
        filtered_results = []
        
        for candidate in candidates:
            # Calculate initial match scores
            skills_match = self._evaluate_skills_match(job, candidate)
            experience_match = self._evaluate_experience_match(job, candidate)
            cv_relevance = self._evaluate_cv_relevance(job, candidate)
            location_match = self._evaluate_location_match(job, candidate)
            
            # Calculate weighted initial score
            weights = {
                "skills_match": 0.5,  # 50% weight on skills
                "experience_match": 0.3,  # 30% weight on experience
                "cv_relevance": 0.15,  # 15% weight on CV relevance
                "location_match": 0.05  # 5% weight on location
            }
            
            match_data = {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "cv_relevance": cv_relevance,
                "location_match": location_match
            }
            
            initial_score = sum(match_data[key] * weights[key] for key in weights)
            
            # Only keep candidates above the minimum threshold
            if initial_score >= min_match_threshold:
                filtered_results.append({
                    "candidate": candidate,
                    "match_data": {
                        **match_data,
                        "initial_match_score": initial_score
                    }
                })
        
        # Sort by initial match score
        filtered_results.sort(key=lambda x: x["match_data"]["initial_match_score"], reverse=True)
        
        return filtered_results
    
    def _detailed_skill_matching(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, bool]:
        """
        Perform detailed matching of candidate skills against job requirements.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            
        Returns:
            Dictionary mapping skill names to boolean match indicators
        """
        skill_matches = {}
        
        # Get all candidate skills (lowercase for case-insensitive comparison)
        candidate_skills = {skill.get("name", "").lower() for skill in candidate.get("skills", [])}
        
        # Check all required skills
        for skill in job.get("required_skills", []):
            skill_name = skill.get("name", "").lower()
            skill_matches[skill_name] = skill_name in candidate_skills
        
        # Check all preferred skills
        for skill in job.get("preferred_skills", []):
            skill_name = skill.get("name", "").lower()
            if skill_name not in skill_matches:  # Avoid duplicates if a skill is both required and preferred
                skill_matches[skill_name] = skill_name in candidate_skills
        
        return skill_matches
    
    def _generate_match_explanation(self, job: Dict[str, Any], candidate: Dict[str, Any],
                                  required_match_pct: float, preferred_match_pct: float,
                                  experience_match: bool, location_match: bool,
                                  skill_match_details: Dict[str, bool]) -> str:
        """
        Generate a human-readable explanation of the match result.
        
        Args:
            job: Dictionary representation of a JobRequirement
            candidate: Dictionary representation of a CandidateProfile
            required_match_pct: Percentage of required skills matched
            preferred_match_pct: Percentage of preferred skills matched
            experience_match: Whether experience requirements are met
            location_match: Whether location requirements are met
            skill_match_details: Dictionary of skill match details
            
        Returns:
            A string explaining the match
        """
        explanation = []
        
        # Skills summary
        if required_match_pct >= 90:
            explanation.append(f"Excellent match for required skills ({required_match_pct:.0f}%).")
        elif required_match_pct >= 70:
            explanation.append(f"Good match for required skills ({required_match_pct:.0f}%).")
        elif required_match_pct >= 50:
            explanation.append(f"Moderate match for required skills ({required_match_pct:.0f}%).")
        else:
            explanation.append(f"Limited match for required skills ({required_match_pct:.0f}%).")
            
        if preferred_match_pct >= 70:
            explanation.append(f"Strong match for preferred skills ({preferred_match_pct:.0f}%).")
        elif preferred_match_pct >= 40:
            explanation.append(f"Some preferred skills matched ({preferred_match_pct:.0f}%).")
        else:
            explanation.append(f"Few preferred skills matched ({preferred_match_pct:.0f}%).")
        
        # Experience
        min_years = job.get("min_years_experience", 0) or 0
        candidate_years = candidate.get("years_of_experience", 0) or 0
        if experience_match:
            explanation.append(f"Meets experience requirements: {candidate_years} years (requirement: {min_years} years).")
        else:
            explanation.append(f"Below experience requirements: {candidate_years} years (requirement: {min_years} years).")
        
        # Location
        if location_match:
            if job.get("remote_friendly") and candidate.get("remote_preference"):
                explanation.append("Position is remote-friendly, and candidate prefers remote work.")
            elif candidate.get("willing_to_relocate"):
                explanation.append("Candidate is willing to relocate for this position.")
            elif job.get("location") and candidate.get("current_location") and job.get("location", "").lower() in candidate.get("current_location", "").lower():
                explanation.append(f"Location match: Candidate is in {candidate.get('current_location')}.")
            else:
                explanation.append("Location requirements are satisfied.")
        else:
            explanation.append("Location mismatch may require consideration.")
        
        # Key matching skills
        matched_skills = [skill for skill, matched in skill_match_details.items() if matched]
        if matched_skills:
            explanation.append(f"Matched skills: {', '.join(matched_skills[:5])}" + 
                             (f" and {len(matched_skills)-5} more" if len(matched_skills) > 5 else ""))
        
        # Missing key skills (focus on required only)
        required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
        missing_required = [skill for skill in required_skills if not skill_match_details.get(skill, False)]
        if missing_required:
            explanation.append(f"Missing required skills: {', '.join(missing_required)}.")
        
        return " ".join(explanation)
    
    def _rank_candidates_with_detail(self, job: Dict[str, Any], filtered_candidates: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Rank candidates based on comprehensive evaluation.
        
        Args:
            job: Dictionary representation of a JobRequirement
            filtered_candidates: List of dictionaries with filtered candidates and match data
            top_n: Maximum number of candidates to return
            
        Returns:
            List of dictionaries with ranked candidates and detailed match results
        """
        ranked_results = []
        
        for item in filtered_candidates:
            candidate = item["candidate"]
            coord_match_data = item["match_data"]
            
            # Perform detailed skill matching
            skill_match_details = self._detailed_skill_matching(job, candidate)
            
            # Calculate skill match percentages
            required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
            preferred_skills = {skill.get("name", "").lower() for skill in job.get("preferred_skills", [])}
            
            required_matched = sum(1 for skill in required_skills if skill_match_details.get(skill, False))
            preferred_matched = sum(1 for skill in preferred_skills if skill_match_details.get(skill, False))
            
            required_match_pct = (required_matched / len(required_skills) * 100) if required_skills else 100
            preferred_match_pct = (preferred_matched / len(preferred_skills) * 100) if preferred_skills else 100
            
            # Determine experience match
            min_years = job.get("min_years_experience", 0) or 0
            candidate_years = candidate.get("years_of_experience", 0) or 0
            experience_match = candidate_years >= min_years
            
            # Determine location match
            location_match = False
            if job.get("remote_friendly") and candidate.get("remote_preference"):
                location_match = True
            elif not job.get("location") or candidate.get("willing_to_relocate"):
                location_match = True
            elif job.get("location") and candidate.get("current_location"):
                if job.get("location", "").lower() in candidate.get("current_location", "").lower() or \
                   candidate.get("current_location", "").lower() in job.get("location", "").lower():
                    location_match = True
            
            # Calculate overall match score (0-100)
            match_score = (
                (required_match_pct * 0.5) +
                (preferred_match_pct * 0.2) +
                (100 if experience_match else 50) * 0.2 +
                (100 if location_match else 50) * 0.05 +
                (coord_match_data["cv_relevance"] * 100) * 0.05
            )
            
            # Generate explanation
            explanation = self._generate_match_explanation(
                job, 
                candidate,
                required_match_pct,
                preferred_match_pct,
                experience_match,
                location_match,
                skill_match_details
            )
            
            # Create detailed result
            result = {
                "candidate": candidate,
                "name": candidate.get("name", ""),
                "match_score": match_score,
                "required_skills_matched": f"{required_match_pct:.1f}%",
                "preferred_skills_matched": f"{preferred_match_pct:.1f}%",
                "skill_match_details": skill_match_details,
                "experience_match": "Yes" if experience_match else "No",
                "location_match": "Yes" if location_match else "No",
                "explanation": explanation
            }
            
            ranked_results.append(result)
        
        # Sort by match score in descending order
        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Limit to top N results
        return ranked_results[:top_n]
    
    def match_candidates_to_job(
        self, 
        job: JobRequirement, 
        all_candidates: List[CandidateProfile],
        min_match_threshold: float = 0.3,
        top_n: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Match candidates to a job using the multi-agent AutoGen approach.
        
        Args:
            job: The validated job requirement model
            all_candidates: List of all candidate profiles
            min_match_threshold: Minimum match threshold for initial filtering
            top_n: Maximum number of candidates to return
            
        Returns:
            Tuple with (match_results, summary_for_display)
        """
        logger.info(f"Starting candidate matching process for job: {job.title}")
        
        # Convert Pydantic models to dictionaries
        job_dict = job.model_dump()
        candidate_dicts = [c.model_dump() for c in all_candidates]
        
        # Simplified direct approach when dealing with function-only mode
        if not self.config_list or not self.verbose:
            logger.info("Using direct matching approach (bypassing agent conversation)")
            filtered_candidates = self._filter_candidates(job_dict, candidate_dicts, min_match_threshold)
            ranked_candidates = self._rank_candidates_with_detail(job_dict, filtered_candidates, top_n)
            return ranked_candidates, []
        
        # Using direct matching approach - no agent conversation needed
        logger.info("Using direct matching approach (simplified)")
        
        # Fallback to direct matching
        filtered_candidates = self._filter_candidates(job_dict, candidate_dicts, min_match_threshold)
        ranked_candidates = self._rank_candidates_with_detail(job_dict, filtered_candidates, top_n)
        
        return ranked_candidates, []


# If we need LLM configuration, it would go here
def get_llm_config(api_key=None):
    """Get LLM configuration for AutoGen (optional, can be None for function-only agents)."""
    if not api_key:
        return None
        
    return {
        "config_list": [{"model": "gpt-4", "api_key": api_key}],
        "cache_seed": 42
    }


def main():
    """Example usage of the AutoGen framework."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create a mock job and candidates for testing
    from pydantic_models import JobRequirement, CandidateProfile, Skill, ExperienceLevel
    
    job = JobRequirement(
        job_id="TEST-JOB-1",
        title="Senior Python Developer",
        department="Engineering",
        experience_level=ExperienceLevel.SENIOR,
        min_years_experience=5.0,
        required_skills=[
            Skill(name="Python", proficiency=4),
            Skill(name="Django", proficiency=4),
            Skill(name="SQL", proficiency=3)
        ],
        preferred_skills=[
            Skill(name="AWS", proficiency=3),
            Skill(name="Docker", proficiency=3),
            Skill(name="Kubernetes", proficiency=2)
        ],
        location="New York",
        remote_friendly=True,
        description="We are looking for an experienced Python developer to join our engineering team.",
        responsibilities=["Develop backend services", "Optimize performance", "Mentor junior developers"],
        qualifications=["Strong Python skills", "Experience with web frameworks", "Database knowledge"]
    )
    
    candidates = [
        CandidateProfile(
            candidate_id="CAND-1",
            name="Alex Johnson",
            email="alex@example.com",
            current_title="Senior Python Developer",
            skills=[
                Skill(name="Python", proficiency=5, years_experience=7),
                Skill(name="Django", proficiency=4, years_experience=5),
                Skill(name="SQL", proficiency=4, years_experience=6),
                Skill(name="AWS", proficiency=3, years_experience=3),
                Skill(name="Docker", proficiency=3, years_experience=2)
            ],
            years_of_experience=7,
            current_location="New York",
            remote_preference=True,
            cv_content="Experienced Python developer with 7 years of experience. Proficient in Django, SQL, AWS, and Docker. Worked on high-traffic web applications and microservices. Led a team of 5 developers."
        ),
        
        CandidateProfile(
            candidate_id="CAND-2",
            name="Jamie Smith",
            email="jamie@example.com",
            current_title="Python Developer",
            skills=[
                Skill(name="Python", proficiency=4, years_experience=4),
                Skill(name="Django", proficiency=3, years_experience=2),
                Skill(name="SQL", proficiency=3, years_experience=3),
                Skill(name="React", proficiency=4, years_experience=3)
            ],
            years_of_experience=4,
            current_location="Boston",
            willing_to_relocate=True,
            cv_content="Python developer with experience in Django and SQL. Developed web applications using React frontend and Django backend. Familiar with test-driven development and CI/CD pipelines."
        ),
        
        CandidateProfile(
            candidate_id="CAND-3",
            name="Taylor Brown",
            email="taylor@example.com",
            current_title="Junior Web Developer",
            skills=[
                Skill(name="Python", proficiency=3, years_experience=2),
                Skill(name="JavaScript", proficiency=4, years_experience=3),
                Skill(name="HTML/CSS", proficiency=4, years_experience=3)
            ],
            years_of_experience=3,
            current_location="Chicago",
            remote_preference=True,
            cv_content="Web developer with focus on frontend technologies. Some experience with Python for scripting tasks. Mainly worked with JavaScript, React, and Node.js."
        )
    ]
    
    # Initialize the AutoGen-based talent matcher
    matcher = AutoGenTalentMatcher(config_list=None, verbose=True)
    
    # Match candidates to job
    ranked_candidates, summary = matcher.match_candidates_to_job(job, candidates, min_match_threshold=0.3, top_n=2)
    
    # Display results
    print("\nTop matched candidates:")
    for i, match in enumerate(ranked_candidates, 1):
        print(f"{i}. {match['name']} - Match score: {match['match_score']:.1f}")
        print(f"   Required skills: {match['required_skills_matched']}")
        print(f"   Experience match: {match['experience_match']}")
        print(f"   Location match: {match['location_match']}")
        print(f"   {match['explanation']}")
        print()

if __name__ == "__main__":
    main()