"""
AutoGen-based Multi-Agent Framework for Talent Matching Tool

This module implements a simplified talent matching framework that:
1. First filters candidates based on job preferences (Column F)
2. Analyzes job requirements in detail
3. Evaluates candidates against those requirements
4. Provides detailed matching explanations and rankings
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
    Talent Matching framework using Microsoft's AutoGen for function execution.
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
        
        # Define related job titles mapping
        self.related_job_titles = {
            "software engineer": ["software developer", "programmer", "coder", "software engineering", 
                                 "web developer", "fullstack", "full stack", "full-stack", 
                                 "backend", "back end", "back-end", "frontend", "front end", "front-end",
                                 "app developer", "application developer", "systems engineer", 
                                 "devops engineer", "cloud engineer", "application engineer"],
            "data scientist": ["data analyst", "data engineer", "machine learning", "ai engineer",
                              "business intelligence", "bi analyst", "data science", "ml engineer", 
                              "statistical analyst", "analytics", "big data", "data mining"],
            "product manager": ["product owner", "program manager", "project manager", "product management",
                               "technical product manager", "product lead"],
            "designer": ["ui designer", "ux designer", "ui/ux", "graphic designer", "web designer",
                         "interaction designer", "visual designer", "product designer"],
            "marketing": ["digital marketing", "marketing specialist", "content marketing", "seo",
                         "social media", "brand", "growth", "marketing manager"],
            "sales": ["account executive", "sales representative", "business development", "account manager"],
            "hr": ["human resources", "talent acquisition", "recruiter", "people operations", 
                  "hr specialist", "human resource"]
        }
        
        # Initialize function caller
        self._setup_function_caller()
        
        logger.info("AutoGen Talent Matcher initialized")
    
    def _setup_function_caller(self):
        """Set up the function caller framework."""
        logger.info("Setting up function-based matching pipeline")
        
        self.function_caller = autogen.UserProxyAgent(
            name="TalentMatchingSystem",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False}
        )
        
        self._register_functions()
    
    def _register_functions(self):
        """Register functions that can be called."""
        function_map = {
            "filter_candidates_by_job_preference": self._filter_candidates_by_job_preference,
            "filter_candidates": self._filter_candidates,
            "rank_candidates": self._rank_candidates_with_detail,
            "evaluate_candidates": self._evaluate_candidates_for_job,
            "extract_candidate_info": self._extract_candidate_info,
            "evaluate_skills_match": self._evaluate_skills_match,
            "evaluate_experience_match": self._evaluate_experience_match,
            "evaluate_cv_relevance": self._evaluate_candidate_cv,
            "evaluate_location_match": self._evaluate_location_match,
            "generate_match_explanation": self._generate_match_explanation
        }
        
        self.function_caller.register_function(function_map=function_map)
    
    def _extract_candidate_info(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from a candidate profile for easier matching."""
        skills = [skill.get("name", "").lower() for skill in candidate.get("skills", [])]
        
        years_experience = candidate.get("years_of_experience", 0)
        if isinstance(years_experience, str):
            years_match = re.search(r'(\d+)', years_experience)
            if years_match:
                years_experience = float(years_match.group(1))
            else:
                years_experience = 0
        
        job_preference = candidate.get("position_preference", "") or candidate.get("jobs_applying_for", "")
        
        return {
            "name": candidate.get("name", ""),
            "skills": skills,
            "years_experience": years_experience,
            "current_title": candidate.get("current_title", ""),
            "current_location": candidate.get("current_location", ""),
            "remote_preference": candidate.get("remote_preference", False),
            "willing_to_relocate": candidate.get("willing_to_relocate", False),
            "cv_content_length": len(candidate.get("cv_content", "") or ""),
            "has_cv": bool(candidate.get("cv_content", "")),
            "job_preference": job_preference
        }
    
    def _are_job_titles_related(self, job_title: str, candidate_preference: str) -> Tuple[bool, float]:
        """Check if a job title is related to a candidate's job preference."""
        if not job_title or not candidate_preference:
            return False, 0.0
            
        job_title = job_title.lower().strip()
        candidate_preference = candidate_preference.lower().strip()
        
        if job_title in candidate_preference or candidate_preference in job_title:
            return True, 1.0
            
        job_key = next((key for key in self.related_job_titles.keys() if key in job_title), None)
        if job_key:
            related_titles = self.related_job_titles[job_key]
            for related in related_titles:
                if related in candidate_preference:
                    return True, 0.9
        
        for key, related_titles in self.related_job_titles.items():
            if key in candidate_preference:
                if any(related in job_title for related in related_titles + [key]):
                    return True, 0.8
            
            if any(related in candidate_preference for related in related_titles):
                if key in job_title or any(related in job_title for related in related_titles):
                    return True, 0.7
        
        job_tokens = set(re.findall(r'\b\w+\b', job_title))
        pref_tokens = set(re.findall(r'\b\w+\b', candidate_preference))
        
        common_tokens = job_tokens.intersection(pref_tokens)
        if common_tokens:
            similarity = len(common_tokens) / len(job_tokens.union(pref_tokens))
            if similarity >= 0.3:
                return True, similarity
        
        return False, 0.0
    
    def _filter_candidates_by_job_preference(self, job: Dict[str, Any], 
                                          candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates based on job preference (Column F)."""
        job_title = job.get("title", "").lower()
        
        print("\nSTEP 1: COORDINATOR AGENT - INITIAL JOB PREFERENCE FILTERING")
        print("-"*80)
        print(f"Filtering candidates based on job preferences for: {job_title}")
        
        filtered_candidates = []
        
        for candidate in candidates:
            candidate_name = candidate.get("name", "Unknown")
            job_preference = candidate.get("position_preference", "") or candidate.get("jobs_applying_for", "")
            
            if not job_preference:
                print(f"  {candidate_name}: No job preference specified - SKIP")
                continue
                
            is_related, similarity = self._are_job_titles_related(job_title, job_preference)
            
            if is_related:
                print(f"  {candidate_name}: Preference '{job_preference}' MATCHES job '{job_title}' (similarity: {similarity:.2f})")
                filtered_candidates.append(candidate)
            else:
                print(f"  {candidate_name}: Preference '{job_preference}' does NOT match job '{job_title}' - SKIP")
        
        print(f"\nJob preference filtering complete:")
        print(f"  {len(filtered_candidates)} candidates match the job preference out of {len(candidates)} total")
        if filtered_candidates:
            print("\nCandidates matching job preference:")
            for i, candidate in enumerate(filtered_candidates, 1):
                print(f"  {i}. {candidate.get('name', 'Unknown')}")
        
        return filtered_candidates
    def _research_role_requirements(self, job_title: str, important_qualities: str) -> Dict[str, Any]:
        """Research and analyze role requirements based on job title and important qualities."""
        expertise = {
            "required_evidence": [],
            "key_indicators": [],
            "project_qualities": [],
            "leadership_requirements": []
        }
        
        title_lower = job_title.lower()
        is_senior = "senior" in title_lower or "lead" in title_lower
        is_mid = "mid" in title_lower or not (is_senior or "junior" in title_lower)
        
        if important_qualities:
            qualities = [q.strip() for q in important_qualities.split(',')]
            expertise["required_evidence"].extend(qualities)
        
        if is_senior:
            expertise["required_evidence"].extend([
                "system architecture",
                "team leadership",
                "project management",
                "mentoring",
                "technical leadership"
            ])
            expertise["project_qualities"].extend([
                "large scale",
                "complex systems",
                "cross-functional",
                "high impact"
            ])
        elif is_mid:
            expertise["required_evidence"].extend([
                "project ownership",
                "technical decision making",
                "team collaboration"
            ])
            expertise["project_qualities"].extend([
                "feature development",
                "system improvements",
                "technical implementation"
            ])
        
        if "engineer" in title_lower or "developer" in title_lower:
            expertise["key_indicators"].extend([
                "code quality",
                "system design",
                "performance optimization",
                "technical documentation",
                "testing methodologies"
            ])
        elif "data" in title_lower:
            expertise["key_indicators"].extend([
                "data analysis",
                "statistical modeling",
                "data pipelines",
                "machine learning",
                "data visualization"
            ])
        
        return expertise

    def _analyze_cv_content(self, cv_content: str, role_expertise: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CV content against role expertise requirements."""
        analysis = {
            "evidence_found": [],
            "evidence_scores": {},
            "project_evaluation": [],
            "leadership_indicators": [],
            "overall_cv_score": 0.0
        }
        
        if not cv_content:
            return analysis
        
        cv_lower = cv_content.lower()
        
        for evidence in role_expertise["required_evidence"]:
            evidence_lower = evidence.lower()
            pattern = f"\\b{evidence_lower}\\w*\\b"
            matches = re.findall(pattern, cv_lower)
            sentences_with_evidence = re.findall(f"[^.]*{pattern}[^.]*\\.", cv_lower)
            
            if matches or sentences_with_evidence:
                analysis["evidence_found"].append(evidence)
                base_score = len(matches) * 0.2
                context_score = len(sentences_with_evidence) * 0.3
                analysis["evidence_scores"][evidence] = min(1.0, base_score + context_score)
        
        project_sections = re.split(r'projects?:|work\s+experience:|experience:', cv_lower)[1:]
        
        for section in project_sections:
            project_score = {
                "indicators_found": [],
                "quality_indicators": [],
                "relevance_score": 0.0
            }
            
            for indicator in role_expertise["key_indicators"]:
                if indicator.lower() in section:
                    project_score["indicators_found"].append(indicator)
            
            for quality in role_expertise["project_qualities"]:
                if quality.lower() in section:
                    project_score["quality_indicators"].append(quality)
            
            if project_score["indicators_found"] or project_score["quality_indicators"]:
                indicator_score = len(project_score["indicators_found"]) / len(role_expertise["key_indicators"])
                quality_score = len(project_score["quality_indicators"]) / len(role_expertise["project_qualities"])
                project_score["relevance_score"] = (indicator_score * 0.6) + (quality_score * 0.4)
                analysis["project_evaluation"].append(project_score)
        
        if analysis["evidence_scores"]:
            evidence_score = sum(analysis["evidence_scores"].values()) / len(role_expertise["required_evidence"])
            project_score = max([p["relevance_score"] for p in analysis["project_evaluation"]], default=0)
            analysis["overall_cv_score"] = (evidence_score * 0.6) + (project_score * 0.4)
        
        return analysis
    def _evaluate_candidate_cv(self, candidate: Dict[str, Any], role_expertise: Dict[str, Any]) -> float:
        """Evaluate a candidate's CV content against role requirements."""
        cv_content = candidate.get("cv_content", "")
        if not cv_content:
            return 0.1
        
        cv_analysis = self._analyze_cv_content(cv_content, role_expertise)
        candidate["cv_analysis"] = cv_analysis
        
        return cv_analysis["overall_cv_score"]
    
    def _evaluate_skills_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Evaluate how well a candidate's skills match the job requirements."""
        required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
        preferred_skills = {skill.get("name", "").lower() for skill in job.get("preferred_skills", [])}
        candidate_skills = {skill.get("name", "").lower() for skill in candidate.get("skills", [])}
        
        required_matched = required_skills.intersection(candidate_skills)
        preferred_matched = preferred_skills.intersection(candidate_skills)
        
        required_score = len(required_matched) / len(required_skills) if required_skills else 1.0
        preferred_score = len(preferred_matched) / len(preferred_skills) if preferred_skills else 1.0
        
        return (required_score * 0.7) + (preferred_score * 0.3)
    
    def _evaluate_experience_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Evaluate how well a candidate's experience matches the job requirements."""
        min_years = job.get("min_years_experience", 0) or 0
        candidate_years = candidate.get("years_of_experience", 0) or 0
        
        print(f"Experience check for {candidate.get('name')}: {candidate_years} years vs required {min_years}")
        
        if candidate_years >= min_years:
            years_score = 1.0
        elif min_years > 0:
            years_score = candidate_years / min_years
        else:
            years_score = 1.0
        
        level_score = 1.0
        if job.get("experience_level") and candidate.get("current_title"):
            if job.get("experience_level", "").lower() in candidate.get("current_title", "").lower():
                level_score = 1.0
            else:
                level_score = 0.5
        
        return (years_score * 0.7) + (level_score * 0.3)
    
    def _evaluate_location_match(self, job: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Evaluate if a candidate's location matches the job location."""
        if job.get("remote_friendly") and candidate.get("remote_preference"):
            return 1.0
            
        if not job.get("location") or candidate.get("willing_to_relocate"):
            return 1.0
            
        if job.get("location") and candidate.get("current_location"):
            if job.get("location", "").lower() in candidate.get("current_location", "").lower() or \
               candidate.get("current_location", "").lower() in job.get("location", "").lower():
                return 1.0
            else:
                return 0.3
                
        return 0.5
    
    def _filter_candidates(self, job: Dict[str, Any], candidates: List[Dict[str, Any]], 
                         min_match_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Filter candidates based on initial criteria."""
        role_expertise = self._research_role_requirements(
            job.get("title", ""),
            job.get("important_qualities", "")
        )
        
        filtered_results = []
        print(f"\nEvaluating {len(candidates)} candidates for {job.get('title', '')}...")
        
        for candidate in candidates:
            candidate_name = candidate.get("name", "Unknown")
            
            skills_match = self._evaluate_skills_match(job, candidate)
            experience_match = self._evaluate_experience_match(job, candidate)
            cv_match = self._evaluate_candidate_cv(candidate, role_expertise)
            location_match = self._evaluate_location_match(job, candidate)
            
            weights = {
                "skills_match": 0.3,
                "experience_match": 0.2,
                "cv_match": 0.4,
                "location_match": 0.1
            }
            
            match_data = {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "cv_match": cv_match,
                "location_match": location_match
            }
            
            initial_score = sum(match_data[key] * weights[key] for key in weights)
            
            print(f"\nCandidate: {candidate_name}")
            print(f"  Skills Match: {skills_match:.2f}")
            print(f"  Experience Match: {experience_match:.2f}")
            print(f"  CV Content Match: {cv_match:.2f}")
            print(f"  Location Match: {location_match:.2f}")
            print(f"  Initial Score: {initial_score:.2f}")
            
            if initial_score >= min_match_threshold:
                print(f"  MATCH: Score {initial_score:.2f} >= threshold {min_match_threshold}")
                filtered_results.append({
                    "candidate": candidate,
                    "match_data": {
                        **match_data,
                        "initial_match_score": initial_score,
                        "role_expertise": role_expertise
                    }
                })
            else:
                print(f"  NO MATCH: Score {initial_score:.2f} < threshold {min_match_threshold}")
        
        filtered_results.sort(key=lambda x: x["match_data"]["initial_match_score"], reverse=True)
        return filtered_results

    def _rank_candidates_with_detail(self, job: Dict[str, Any], filtered_candidates: List[Dict[str, Any]], 
                                   top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank candidates based on comprehensive evaluation."""
        ranked_results = []
        
        print(f"\nPerforming detailed ranking of {len(filtered_candidates)} candidates...")
        
        for item in filtered_candidates:
            candidate = item["candidate"]
            candidate_name = candidate.get("name", "Unknown")
            coord_match_data = item["match_data"]
            
            print(f"\nDetailed evaluation for candidate: {candidate_name}")
            
            skill_match_details = self._detailed_skill_matching(job, candidate)
            
            required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
            preferred_skills = {skill.get("name", "").lower() for skill in job.get("preferred_skills", [])}
            
            required_matched = sum(1 for skill in required_skills if skill_match_details.get(skill, False))
            preferred_matched = sum(1 for skill in preferred_skills if skill_match_details.get(skill, False))
            
            required_match_pct = (required_matched / len(required_skills) * 100) if required_skills else 100
            preferred_match_pct = (preferred_matched / len(preferred_skills) * 100) if preferred_skills else 100
            
            print(f"  Required Skills: {required_matched}/{len(required_skills)} = {required_match_pct:.1f}%")
            print(f"  Preferred Skills: {preferred_matched}/{len(preferred_skills)} = {preferred_match_pct:.1f}%")
            
            min_years = job.get("min_years_experience", 0) or 0
            candidate_years = candidate.get("years_of_experience", 0) or 0
            experience_match = candidate_years >= min_years
            
            print(f"  Experience: {candidate_years} years vs. required {min_years} = {experience_match}")
            
            location_match = False
            if job.get("remote_friendly") and candidate.get("remote_preference"):
                location_match = True
            elif not job.get("location") or candidate.get("willing_to_relocate"):
                location_match = True
            elif job.get("location") and candidate.get("current_location"):
                if job.get("location", "").lower() in candidate.get("current_location", "").lower() or \
                candidate.get("current_location", "").lower() in job.get("location", "").lower():
                    location_match = True
            
            print(f"  Location Match: {location_match}")
            
            match_score = (
                (required_match_pct * 0.5) +
                (preferred_match_pct * 0.2) +
                (100 if experience_match else 50) * 0.2 +
                (100 if location_match else 50) * 0.05 +
                (coord_match_data.get("cv_match", 0) * 100) * 0.05
            )
            
            print(f"  Final Match Score: {match_score:.1f}")
            
            explanation = self._generate_match_explanation(
                job, 
                candidate,
                required_match_pct,
                preferred_match_pct,
                experience_match,
                location_match,
                skill_match_details
            )
            
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
        
        ranked_results.sort(key=lambda x: x["match_score"], reverse=True)
        top_results = ranked_results[:top_n]
        
        print("\nFinal Candidate Ranking:")
        for i, result in enumerate(top_results, 1):
            print(f"  {i}. {result['name']} - Score: {result['match_score']:.1f}")
            print(f"     {result['explanation']}")
        
        return top_results
    
    def _evaluate_candidates_for_job(self, job: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                   top_n: int = 10, min_threshold: float = 0.3) -> Dict[str, Any]:
        """Complete assessment of candidates for a job, simulating agent interaction."""
        print(f"\nEVALUATION PROCESS FOR JOB: {job.get('title', 'Unknown Position')}")
        print("="*80)
        
        job_preference_filtered = self._filter_candidates_by_job_preference(job, candidates)
        
        if not job_preference_filtered:
            print("\nNo candidates match the job preference. Proceeding with all candidates.")
            job_preference_filtered = candidates
        
        print("\nSTEP 2: COORDINATOR AGENT - DETAILED CANDIDATE FILTERING")
        print("-"*80)
        print("Analyzing job requirements and filtering candidates based on detailed criteria...")
        
        filtered_candidates = self._filter_candidates(job, job_preference_filtered, min_threshold)
        
        if not filtered_candidates:
            return {
                "job_title": job.get("title", "Unknown Position"),
                "total_candidates": len(candidates),
                "filtered_candidates": 0,
                "top_candidates": 0,
                "ranked_candidates": []
            }
        
        print("\nSTEP 3: HR MANAGER AGENT - DETAILED CANDIDATE EVALUATION")
        print("-"*80)
        print("Performing detailed analysis and ranking of pre-filtered candidates...")
        
        ranked_candidates = self._rank_candidates_with_detail(job, filtered_candidates, top_n)
        
        result = {
            "job_title": job.get("title", "Unknown Position"),
            "total_candidates": len(candidates),
            "filtered_by_preference": len(job_preference_filtered),
            "filtered_candidates": len(filtered_candidates),
            "top_candidates": len(ranked_candidates),
            "ranked_candidates": ranked_candidates
        }
        
        print("\nEVALUATION COMPLETE")
        print(f"Found {len(ranked_candidates)} suitable candidates out of {len(candidates)} total applicants")
        print(f"First filtered to {len(job_preference_filtered)} candidates by job preference")
        print(f"Then filtered to {len(filtered_candidates)} candidates by detailed criteria")
        print("="*80)
        
        return result
    
    def match_candidates_to_job(self, job: JobRequirement, all_candidates: List[CandidateProfile],
                              min_match_threshold: float = 0.3, top_n: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Match candidates to a job using the enhanced matching framework."""
        logger.info(f"Starting candidate matching process for job: {job.title}")
        
        job_dict = job.model_dump()
        candidate_dicts = [c.model_dump() for c in all_candidates]
        
        result = self._evaluate_candidates_for_job(
            job_dict, 
            candidate_dicts, 
            top_n=top_n, 
            min_threshold=min_match_threshold
        )
        
        return result.get("ranked_candidates", []), []

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
        """Generate a human-readable explanation of the match result."""
        explanation = []
        
        # Get CV analysis if available
        cv_analysis = candidate.get("cv_analysis", {})
        evidence_found = cv_analysis.get("evidence_found", [])
        project_evaluations = cv_analysis.get("project_evaluation", [])
        
        # Skills summary
        if required_match_pct >= 90:
            explanation.append(f"Excellent match for required skills ({required_match_pct:.0f}%).")
        elif required_match_pct >= 70:
            explanation.append(f"Good match for required skills ({required_match_pct:.0f}%).")
        elif required_match_pct >= 50:
            explanation.append(f"Moderate match for required skills ({required_match_pct:.0f}%).")
        else:
            explanation.append(f"Limited match for required skills ({required_match_pct:.0f}%).")
        
        # Experience
        min_years = job.get("min_years_experience", 0) or 0
        candidate_years = candidate.get("years_of_experience", 0) or 0
        if experience_match:
            explanation.append(f"Meets experience requirements with {candidate_years} years (requirement: {min_years} years).")
        else:
            explanation.append(f"Below experience requirement with {candidate_years} years (requirement: {min_years} years).")
        
        # CV content analysis
        if evidence_found:
            explanation.append(f"Demonstrated expertise in: {', '.join(evidence_found[:3])}.")
        
        if project_evaluations:
            relevant_projects = sum(1 for p in project_evaluations if p["relevance_score"] > 0.5)
            if relevant_projects > 0:
                explanation.append(f"Has {relevant_projects} relevant projects showing required capabilities.")
        
        # Location
        if location_match:
            if job.get("remote_friendly") and candidate.get("remote_preference"):
                explanation.append("Position is remote-friendly, and candidate prefers remote work.")
            elif candidate.get("willing_to_relocate"):
                explanation.append("Candidate is willing to relocate for this position.")
            elif job.get("location") and candidate.get("current_location"):
                explanation.append(f"Location match: Candidate is in {candidate.get('current_location')}.")
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


# Helper function for LLM configuration
def get_llm_config(api_key=None):
    """Get LLM configuration for AutoGen (optional, can be None for function-only agents)."""
    if not api_key:
        return None
        
    return {
        "config_list": [{"model": "gpt-4", "api_key": api_key}],
        "cache_seed": 42
    }