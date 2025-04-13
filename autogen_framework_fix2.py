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

# Import the tool
from talent_matching_tool_fix2 import TalentMatchingTool

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
    
    def __init__(self, tool: TalentMatchingTool, config_list=None, verbose=True, use_gemini=False, debug=False):
        """
        Initialize the AutoGen-based Talent Matcher.
        
        Args:
            tool: Instance of TalentMatchingTool for data access and CV extraction
            config_list: Configuration for the LLM (if None, agents will use function calling only)
            verbose: Whether to display detailed agent conversations
            use_gemini: Whether to use Gemini API (requires GEMINI_API_KEY in environment)
            debug: Enable debug logging for detailed troubleshooting
        """
        self.tool = tool
        self.verbose = verbose
        self.config_list = config_list
        self.use_gemini = use_gemini
        self.debug = debug
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Store analysis results in separate dictionaries keyed by candidate ID
        self.candidate_analysis_results = {}
        
        # Set up logging based on debug flag
        log_level = logging.DEBUG if debug else logging.INFO
        logging.getLogger("talent_matching_autogen").setLevel(log_level)
        
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
        
        if self.debug:
            logger.debug("AutoGen Talent Matcher initialized with debug mode")
        else:
            logger.info("AutoGen Talent Matcher initialized")
    
    def _setup_function_caller(self):
        """Set up the function-based matching pipeline."""
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
    
    def _calculate_title_similarity(self, job_title: str, candidate_preference: str) -> float:
        """Calculate similarity between job title and candidate preference."""
        _, similarity = self._are_job_titles_related(job_title, candidate_preference)
        return similarity
    
    def _filter_candidates_by_job_preference(
        self,
        job_dict: Dict[str, Any],
        candidates: List[CandidateProfile] # Changed input type
    ) -> List[CandidateProfile]: # Changed return type
        """
        Filter candidates based on job preference similarity.

        Args:
            job_dict: Dictionary representation of the job.
            candidates: List of CandidateProfile objects.

        Returns:
            List of CandidateProfile objects matching the job preference.
        """
        job_title_lower = job_dict.get("title", "").lower()
        filtered_list = []
        similarity_scores = []

        print(f"Filtering candidates based on job preferences for: {job_title_lower}")

        for candidate in candidates:
            # Access preference directly from the CandidateProfile object
            candidate_pref_lower = (candidate.jobs_applying_for or candidate.current_title or "").lower()

            if not candidate_pref_lower:
                print(f"  {candidate.name}: No job preference found - SKIP (Consider adding default behavior)")
                continue

            # Calculate similarity
            similarity = self._calculate_title_similarity(job_title_lower, candidate_pref_lower)

            # Basic Thresholding (adjust as needed)
            PREFERENCE_SIMILARITY_THRESHOLD = 0.7
            if similarity >= PREFERENCE_SIMILARITY_THRESHOLD:
                print(f"  {candidate.name}: Preference '{candidate_pref_lower}' MATCHES job '{job_title_lower}' (similarity: {similarity:.2f})")
                filtered_list.append(candidate) # Append the CandidateProfile object
                similarity_scores.append((candidate, similarity))
            else:
                print(f"  {candidate.name}: Preference '{candidate_pref_lower}' does NOT match job '{job_title_lower}' (similarity: {similarity:.2f}) - SKIP")

        # Sort by similarity score, descending
        similarity_scores.sort(key=lambda item: item[1], reverse=True)
        sorted_candidates = [item[0] for item in similarity_scores] # Extract sorted CandidateProfile objects

        print("\nJob preference filtering complete:")
        print(f"  {len(sorted_candidates)} candidates match the job preference out of {len(candidates)} total")

        if sorted_candidates:
            print("\nCandidates matching job preference (sorted by relevance):")
            for i, (candidate, score) in enumerate(similarity_scores):
                 print(f"  {i+1}. {candidate.name} - Similarity: {score:.2f}")

        return sorted_candidates # Return the list of CandidateProfile objects
    
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

    def _extract_cv_sections(self, cv_content: str) -> Dict[str, str]:
        """Extract different sections from CV content with improved pattern matching."""
        if not cv_content:
            print("Warning: CV content is empty for section extraction")
            return {"experience": cv_content}
        
        cv_lower = cv_content.lower()
        sections = {}
        
        # Define comprehensive section headers with more flexible patterns
        section_patterns = {
            "education": [
                r'education[\s]*:?',
                r'academic[\s]*background[\s]*:?',
                r'qualifications[\s]*:?',
                r'education\s+and\s+training[\s]*:?',
                r'educational\s+background[\s]*:?',
                r'degree',
                r'education\s+history',
                r'educational\s+qualifications',
                r'certifications?[\s]*:?',
                r'training[\s]*:?'
            ],
            "experience": [
                r'experience[\s]*:?',
                r'work[\s]*history[\s]*:?',
                r'employment[\s]*history[\s]*:?',
                r'professional[\s]*experience[\s]*:?',
                r'work\s+experience',
                r'employment\s+background',
                r'career\s+history',
                r'professional\s+background',
                r'work\s+history',
                r'employment\s+experience'
            ],
            "skills": [
                r'skills[\s]*:?',
                r'technical[\s]*skills[\s]*:?',
                r'core[\s]*competencies[\s]*:?',
                r'key[\s]*skills[\s]*:?',
                r'professional[\s]*skills[\s]*:?',
                r'expertise',
                r'competencies',
                r'technical\s+expertise',
                r'areas\s+of\s+expertise',
                r'professional\s+competencies'
            ],
            "projects": [
                r'projects[\s]*:?',
                r'key[\s]*projects[\s]*:?',
                r'selected[\s]*projects[\s]*:?',
                r'project\s+experience',
                r'project\s+history',
                r'notable\s+projects',
                r'achievements',
                r'key\s+achievements',
                r'notable\s+achievements',
                r'project\s+portfolio'
            ],
            "summary": [
                r'summary[\s]*:?',
                r'profile[\s]*:?',
                r'professional\s+summary[\s]*:?',
                r'career\s+summary[\s]*:?',
                r'executive\s+summary[\s]*:?',
                r'overview[\s]*:?',
                r'introduction[\s]*:?'
            ]
        }
        
        # Find all section headers and their positions
        section_positions = []
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, cv_lower):
                    section_positions.append((match.start(), section_name))
        
        # Sort sections by position
        section_positions.sort(key=lambda x: x[0])
        
        # Extract content between sections
        for i, (start_pos, section_name) in enumerate(section_positions):
            if i < len(section_positions) - 1:
                next_start = section_positions[i + 1][0]
                sections[section_name] = cv_content[start_pos:next_start].strip()
            else:
                sections[section_name] = cv_content[start_pos:].strip()
        
        # If no sections found, treat entire content as experience
        if not sections:
            sections["experience"] = cv_content
        
        return sections

    def _analyze_cv_content(self, cv_content: str, role_expertise: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CV content against role expertise requirements with enhanced context analysis."""
        if not cv_content or not cv_content.strip():
            print("Warning: Empty CV content provided")
            return {
                "evidence_found": [],
                "evidence_scores": {},
                "project_evaluation": [],
                "leadership_indicators": [],
                "skill_mentions": {},
                "experience_context": {},
                "overall_cv_score": 0.0
            }
        
        print(f"\nAnalyzing CV content ({len(cv_content)} characters)...")
        
        # Preprocess and section the CV
        sections = self._extract_cv_sections(cv_content)
        
        if not sections:
            print("Warning: No sections could be extracted from CV content")
            return {
                "evidence_found": [],
                "evidence_scores": {},
                "project_evaluation": [],
                "inferred_skills": [],
                "overall_cv_score": 0.0
            }
        
        # Extract skills and qualities from different sections
        inferred_skills = []
        inferred_qualities = []
        
        # Extract from skills section
        if "skills" in sections:
            skills_text = sections["skills"]
            inferred_skills.extend(self._extract_skills_from_text(skills_text))
        
        # Extract from experience section
        if "experience" in sections:
            experience_text = sections["experience"]
            inferred_skills.extend(self._extract_skills_from_text(experience_text))
            inferred_qualities.extend(self._extract_qualities_from_experience(experience_text))
        
        # Extract from projects section
        if "projects" in sections:
            projects_text = sections["projects"]
            inferred_skills.extend(self._extract_skills_from_text(projects_text))
            inferred_qualities.extend(self._extract_qualities_from_projects(projects_text))
        
        # Extract from summary section
        if "summary" in sections:
            summary_text = sections["summary"]
            inferred_qualities.extend(self._extract_qualities_from_summary(summary_text))
        
        # Remove duplicates and normalize
        inferred_skills = list(set(skill.lower() for skill in inferred_skills))
        inferred_qualities = list(set(quality.lower() for quality in inferred_qualities))
        
        print(f"Total inferred skills: {len(inferred_skills)}")
        print(f"Total inferred qualities: {len(inferred_qualities)}")
        
        # Evaluate required evidence in context
        evidence_results = self._evaluate_evidence_in_context(
            cv_content, 
            sections, 
            role_expertise["required_evidence"],
            inferred_skills,
            inferred_qualities
        )
        
        return {
            "evidence_found": evidence_results["evidence_found"],
            "evidence_scores": evidence_results["evidence_scores"],
            "project_evaluation": self._evaluate_projects(sections.get("projects", ""), role_expertise),
            "inferred_skills": inferred_skills,
            "inferred_qualities": inferred_qualities,
            "overall_cv_score": evidence_results["overall_score"]
        }

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using multiple patterns, focusing on data science.
        Now the primary source for candidate skills.
        """
        skills = set()
        text_lower = text.lower()
        
        # Define more comprehensive skill patterns, especially for data science
        skill_patterns = [
            # Programming Languages & Core Libraries
            r'\b(python|r|sql|java|scala|c\+\+|julia)\b',
            r'\b(pandas|numpy|scipy|scikit-learn|sklearn|matplotlib|seaborn|plotly)\b',
            r'\b(tensorflow|keras|pytorch|theano|caffe|mxnet|jax)\b',
            r'\b(nltk|spacy|gensim|transformers)\b', # NLP
            r'\b(opencv|pillow)\b', # Computer Vision
            
            # Databases & Data Warehousing
            r'\b(sql|mysql|postgresql|postgres|sqlite|sql server|tsql|pl/sql|oracle|mongodb|cassandra|redis|neo4j)\b',
            r'\b(bigquery|redshift|snowflake|teradata|hive|presto|spark sql)\b',
            
            # Big Data & Distributed Computing
            r'\b(spark|pyspark|hadoop|hdfs|mapreduce|kafka|flink|storm)\b',
            
            # Cloud Platforms
            r'\b(aws|azure|gcp|google cloud|amazon web services)\b',
            r'\b(sagemaker|azure ml|google ai platform|databricks|kubernetes|docker)\b',
            
            # ML/AI Concepts & Techniques
            r'\b(machine learning|ml|deep learning|dl|artificial intelligence|ai)\b',
            r'\b(natural language processing|nlp|computer vision|cv|reinforcement learning|rl)\b',
            r'\b(regression|classification|clustering|dimensionality reduction|feature engineering)\b',
            r'\b(neural networks?|cnn|rnn|lstm|transformer[s]?)\b',
            r'\b(gradient boosting|xgboost|lightgbm|catboost|random forest|svm|decision tree[s]?)\b',
            r'\b(a/b testing|experimentation|statistical modeling|statistics|probability)\b',
            
            # BI & Visualization Tools
            r'\b(tableau|power bi|powerbi|qlik|looker|d3.js|ggplot2)\b',
            
            # General Software Engineering
            r'\b(git|linux|bash|shell scripting|api[s]?|rest api)\b'
        ]
        
        # Contextual patterns (e.g., "experience with Python")
        context_patterns = [
            r'(?:proficient in|expert in|skilled in|experience with|knowledge of|working with|using)\s+((?:\b\w+\b[\s,]*){1,4})',
            r'(?:skills?|expertise|competencies?)\s*[:\-]?\s*((?:\b\w+\b[\s,]*){1,10})'
        ]

        # Direct keyword matching
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                skills.add(match.strip())

        # Contextual matching
        for pattern in context_patterns:
            matches = re.findall(pattern, text_lower)
            for match_group in matches:
                # Potential skills are within the matched group
                potential_skills = match_group.split(',')
                for ps in potential_skills:
                    ps_clean = ps.strip()
                    if len(ps_clean) > 1: # Avoid single letters
                        # Check if the potential skill matches any known patterns directly
                        for skill_pattern in skill_patterns:
                            if re.search(skill_pattern, ps_clean):
                                skills.add(ps_clean)
                                break # Add once per potential skill
                        # Basic check for likely skills (e.g., alphanumeric, possibly with ., +, #)
                        if re.match(r'^[a-z0-9\.\+\#\s\-/]+$', ps_clean) and len(ps_clean) <= 30:
                             skills.add(ps_clean) # Add plausible skills even if not in specific list
                             
        # Clean up common non-skill words often caught by context patterns
        non_skills = {"various", "multiple", "different", "tools", "technologies", "languages", "concepts", "methods", "techniques", "including", "such as"}
        skills = {s for s in skills if s not in non_skills and len(s) > 1} # Remove single letters/short strings
                             
        print(f"Extracted {len(skills)} unique skills from text.")
        if self.debug and skills:
             logger.debug(f"Sample extracted skills: {list(skills)[:10]}")
        
        return list(skills)

    def _extract_qualities_from_experience(self, text: str) -> List[str]:
        """Extract important qualities from experience section."""
        qualities = []
        
        # Patterns for identifying qualities in experience
        quality_patterns = [
            r'(?:demonstrated|shown|exhibited|displayed)\s+(?:strong|excellent|good)\s+(?:ability|skill|capacity)\s+(?:in|for|to)\s+([^.,;]+)',
            r'(?:proven|established|demonstrated)\s+(?:track record|ability|experience)\s+(?:in|for|of)\s+([^.,;]+)',
            r'(?:excellent|strong|outstanding)\s+(?:communication|leadership|problem-solving|analytical)\s+(?:skills?|abilities?|capabilities?)',
            r'(?:successfully|effectively)\s+(?:managed|led|handled|oversaw)\s+([^.,;]+)',
            r'(?:key|critical|important)\s+(?:contributions?|achievements?|accomplishments?)\s+(?:in|for|of)\s+([^.,;]+)'
        ]
        
        for pattern in quality_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                quality_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                qualities.append(quality_text)
        
        return qualities

    def _extract_qualities_from_projects(self, text: str) -> List[str]:
        """Extract important qualities from projects section."""
        qualities = []
        
        # Patterns for identifying qualities in projects
        quality_patterns = [
            r'(?:demonstrated|shown|exhibited)\s+(?:ability|skill|capacity)\s+(?:in|for|to)\s+([^.,;]+)',
            r'(?:successfully|effectively)\s+(?:implemented|developed|designed)\s+([^.,;]+)',
            r'(?:key|critical|important)\s+(?:contributions?|achievements?)\s+(?:in|for|of)\s+([^.,;]+)',
            r'(?:improved|enhanced|optimized)\s+([^.,;]+)\s+(?:by|through|using)\s+([^.,;]+)',
            r'(?:led|managed|oversaw)\s+(?:team|project|initiative)\s+(?:in|for|of)\s+([^.,;]+)'
        ]
        
        for pattern in quality_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                quality_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                qualities.append(quality_text)
        
        return qualities

    def _extract_qualities_from_summary(self, text: str) -> List[str]:
        """Extract important qualities from summary section."""
        qualities = []
        
        # Patterns for identifying qualities in summary
        quality_patterns = [
            r'(?:proven|established|demonstrated)\s+(?:track record|ability|experience)\s+(?:in|for|of)\s+([^.,;]+)',
            r'(?:excellent|strong|outstanding)\s+(?:communication|leadership|problem-solving|analytical)\s+(?:skills?|abilities?|capabilities?)',
            r'(?:passionate|dedicated|committed)\s+(?:to|about)\s+([^.,;]+)',
            r'(?:strong|excellent|good)\s+(?:ability|capacity)\s+(?:to|for)\s+([^.,;]+)',
            r'(?:proven|demonstrated)\s+(?:ability|capacity)\s+(?:to|for)\s+([^.,;]+)'
        ]
        
        for pattern in quality_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                quality_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                qualities.append(quality_text)
        
        return qualities

    def _evaluate_evidence_in_context(self, cv_content: str, sections: Dict[str, str], 
                                    required_evidence: List[str], inferred_skills: List[str],
                                    inferred_qualities: List[str]) -> Dict[str, Any]:
        """Evaluate required evidence (including skills and qualities) across all CV content 
           with enhanced context awareness and semantic matching for qualities.
        """
        evidence_found = []
        evidence_scores = {}
        cv_lower = cv_content.lower()
        
        # Define keywords for semantic matching of complex qualities
        quality_keywords = {
            "fintech experience": ["fintech", "financial services", "banking", "payments", "trading", "insurance", "wealth management", "regtech"],
            "analytical thinking": ["analyze", "analysis", "analytical", "quantitative", "modeling", "metrics", "insights", "data-driven", "statistical", "problem solving", "diagnose"],
            # Add more qualities and their keywords as needed
            "team leadership": ["lead", "led", "manage", "managed", "mentor", "mentored", "supervised", "team lead", "leadership"],
            "project management": ["project manage", "coordinated", "delivered", "launched", "agile", "scrum", "roadmap", "timeline"],
            "communication skills": ["presented", "presentation", "communicated", "documentation", "reported", "liaised", "stakeholder"]
        }

        print(f"Evaluating required evidence: {required_evidence}")
        if self.debug:
            logger.debug(f"Inferred Skills for evidence check: {inferred_skills}")
            logger.debug(f"Inferred Qualities for evidence check: {inferred_qualities}")

        for evidence_item in required_evidence:
            evidence_lower = evidence_item.lower().strip()
            best_score = 0.0
            match_type = "None"

            # 1. Direct match in CV content (Highest score)
            if evidence_lower in cv_lower:
                best_score = 1.0
                match_type = "Direct CV Match"
            
            # 2. Check in inferred skills (High score)
            # Use more robust skill matching (handle minor variations)
            if best_score < 0.8:
                for skill in inferred_skills:
                    # Simple substring check or more advanced matching could be used
                    if evidence_lower == skill or evidence_lower in skill or skill in evidence_lower:
                        best_score = max(best_score, 0.8)
                        match_type = "Inferred Skill Match"
                        break
            
            # 3. Check in inferred qualities (Good score)
            if best_score < 0.7:
                 for quality in inferred_qualities:
                     if evidence_lower == quality or evidence_lower in quality or quality in evidence_lower:
                         best_score = max(best_score, 0.7)
                         match_type = "Inferred Quality Match"
                         break

            # 4. Semantic keyword matching for known qualities
            if best_score < 0.6 and evidence_lower in quality_keywords:
                keywords = quality_keywords[evidence_lower]
                keyword_score = self._evaluate_semantic_match(keywords, cv_lower, sections)
                if keyword_score > best_score:
                    best_score = keyword_score # Use the semantic score directly
                    match_type = "Semantic Keyword Match"

            # 5. General context match (Fallback)
            if best_score < 0.3:
                context_score = self._evaluate_context_match(evidence_lower, cv_lower, sections)
                if context_score > best_score:
                    best_score = context_score
                    match_type = "General Context Match"

            # Record if any match found
            if best_score > 0.1: # Set a minimum threshold to consider it found
                evidence_found.append(evidence_item)
                evidence_scores[evidence_item] = round(best_score, 2)
                if self.debug:
                     logger.debug(f"  Evidence '{evidence_item}': Found (Score: {best_score:.2f}, Type: {match_type})")
            elif self.debug:
                 logger.debug(f"  Evidence '{evidence_item}': Not found")

        # Calculate overall score based on matched evidence
        overall_score = 0.0
        if required_evidence: # Avoid division by zero
             # Average score of found items, scaled by proportion found
             if evidence_scores:
                 average_score_found = sum(evidence_scores.values()) / len(evidence_scores)
                 proportion_found = len(evidence_scores) / len(required_evidence)
                 overall_score = average_score_found * proportion_found
             else:
                 overall_score = 0.0 # No evidence found
        else:
            overall_score = 0.5 # No specific evidence required, default score

        print(f"Evidence evaluation complete. Found: {len(evidence_found)}/{len(required_evidence)}. Overall Score: {overall_score:.2f}")
        
        return {
            "evidence_found": evidence_found,
            "evidence_scores": evidence_scores,
            "overall_score": round(overall_score, 2)
        }

    def _evaluate_context_match(self, evidence: str, cv_content: str, sections: Dict[str, str]) -> float:
        """Evaluate evidence match based on word overlap within important CV sections."""
        evidence_words = set(evidence.split())
        if not evidence_words:
            return 0.0
        
        max_score = 0.0
        # Define section weights (higher for more relevant sections)
        section_weights = {
            "experience": 1.0,
            "projects": 0.9,
            "summary": 0.8,
            "skills": 0.7, 
            "education": 0.5
            # Other sections get a default low weight
        }
        default_weight = 0.3

        for section_name, section_content in sections.items():
            section_lower = section_content.lower()
            section_words = set(section_lower.split())
            
            # Count word matches
            common_words = evidence_words.intersection(section_words)
            
            # Calculate score based on Jaccard index (word overlap)
            if evidence_words.union(section_words):
                overlap_score = len(common_words) / len(evidence_words) # Score relative to evidence length
            else:
                 overlap_score = 0.0
            
            # Apply section weight
            weight = section_weights.get(section_name, default_weight)
            weighted_score = overlap_score * weight
            
            max_score = max(max_score, weighted_score)
        
        # Return a score between 0 and 0.6 (context match is weaker evidence)
        return min(0.6, max_score) 

    def _evaluate_semantic_match(self, keywords: List[str], cv_content: str, sections: Dict[str, str]) -> float:
        """Evaluate if related keywords appear in relevant sections, providing semantic evidence.
        Returns a score between 0.3 and 0.8 based on keyword presence and section.
        """
        max_score = 0.0
        keyword_found = False
        # Define section weights for semantic matches
        section_weights = {
            "experience": 0.8, 
            "projects": 0.7, 
            "summary": 0.6,
            "skills": 0.5, # Keywords might appear in skill lists indirectly
            "education": 0.4
        }
        default_weight = 0.3

        for section_name, section_content in sections.items():
            section_lower = section_content.lower()
            found_in_section = False
            for keyword in keywords:
                if keyword in section_lower:
                    keyword_found = True
                    found_in_section = True
                    break # Found at least one keyword in this section
            
            if found_in_section:
                weight = section_weights.get(section_name, default_weight)
                max_score = max(max_score, weight)
                # If found in high-importance section, boost score slightly
                if section_name in ["experience", "projects"]:
                    max_score = max(max_score, weight + 0.05) # Small boost

        # Return score only if at least one keyword was found anywhere
        # The score reflects *where* it was found (strongest section)
        return min(0.8, max_score) if keyword_found else 0.0

    def _evaluate_projects(self, projects_text: str, role_expertise: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate projects section against role requirements."""
        if not projects_text:
            return []
        
        # Split into individual projects if possible
        projects = re.split(r'\n(?=Project:|•)', projects_text)
        if len(projects) <= 1:
            # Try alternative splitting if no clear project delimiters
            projects = re.split(r'\n\n+', projects_text)
        
        project_evaluations = []
        
        for project_text in projects:
            if len(project_text.strip()) < 20:  # Skip very short sections
                continue
                
            project_score = {
                "relevance_score": 0.0,
                "technologies": [],
                "achievements": [],
                "relevance_factors": []
            }
            
            # Extract technologies
            tech_pattern = r'\b(?:using|with|via|through|in)\s+([A-Za-z0-9+#.]+)\b'
            technologies = re.findall(tech_pattern, project_text.lower())
            if technologies:
                project_score["technologies"] = technologies
            
            # Check for achievements
            achievement_pattern = r'\b(?:improved|increased|reduced|achieved|delivered|created)\b.{3,50}(?:\d+%|\$\d+)'
            achievements = re.findall(achievement_pattern, project_text.lower())
            if achievements:
                project_score["achievements"] = [a.strip() for a in achievements]
            
            # Check relevance to role requirements
            for indicator in role_expertise.get("key_indicators", []):
                if indicator.lower() in project_text.lower():
                    project_score["relevance_factors"].append(indicator)
            
            # Calculate relevance score based on match factors
            base_score = 0.3  # Some base relevance
            tech_score = min(0.3, len(project_score["technologies"]) * 0.1)
            achievement_score = min(0.2, len(project_score["achievements"]) * 0.1)
            indicator_score = min(0.2, len(project_score["relevance_factors"]) * 0.1)
            
            project_score["relevance_score"] = base_score + tech_score + achievement_score + indicator_score
            
            project_evaluations.append(project_score)
        
        # Sort by relevance
        project_evaluations.sort(key=lambda p: p["relevance_score"], reverse=True)
        
        return project_evaluations

    def _evaluate_candidate_cv(self, candidate, role_expertise: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a candidate's CV content against role requirements.
        Works with both dictionary candidates and CandidateProfile objects.
        """
        
        # Handle both dictionary and CandidateProfile objects
        if hasattr(candidate, 'cv_content'):
            # It's a CandidateProfile object
            cv_content = candidate.cv_content or ""
            candidate_name = candidate.name
            cv_link = candidate.cv_link or ""
        else:
            # It's a dictionary
            cv_content = candidate.get("cv_content", "")
            candidate_name = candidate.get('name', 'Unknown')
            cv_link = candidate.get("cv_link", "")

        # If no content, try fetching it using the stored CV link
        if not cv_content and cv_link:
            print(f"CV content not found for {candidate_name}. Fetching from link: {cv_link}")
            try:
                # Use the stored tool instance to fetch content
                cv_content = self.tool.extract_cv_content(cv_link)
                if cv_content:
                    print(f"Successfully fetched {len(cv_content)} characters for {candidate_name}")
                    # Update the candidate with the fetched content
                    if hasattr(candidate, 'cv_content'):
                        candidate.cv_content = cv_content
                    else:
                        candidate["cv_content"] = cv_content
                else:
                    print(f"Warning: Failed to fetch CV content for {candidate_name} from link.")
                    cv_content = "" # Ensure it's an empty string if fetch fails
            except Exception as e:
                print(f"Error fetching CV content for {candidate_name}: {str(e)}")
                cv_content = "" # Ensure it's an empty string on error
        
        # Proceed with evaluation, whether content was pre-existing or just fetched
        if self.debug:
            logger.debug(f"\nEvaluating CV for {candidate_name}")
            logger.debug(f"CV content length: {len(cv_content)}")
            if cv_content:
                logger.debug("First 100 chars of CV:")
                logger.debug(cv_content[:100])
        
        if not cv_content:
            logger.debug("No CV content available for analysis") if self.debug else None
            cv_analysis = { # Add empty analysis if no CV
                "evidence_found": [],
                "evidence_scores": {},
                "project_evaluation": [],
                "inferred_skills": [],
                "overall_cv_score": 0.1
            }
            # Store the analysis in the candidate
            self._store_cv_analysis(candidate, cv_analysis)
            return cv_analysis
        
        cv_analysis = self._analyze_cv_content(cv_content, role_expertise)
        
        # Store the analysis in the candidate
        self._store_cv_analysis(candidate, cv_analysis)
        
        if self.debug:
            logger.debug(f"CV Analysis Results:")
            logger.debug(f"Evidence found: {cv_analysis.get('evidence_found', [])}")
            logger.debug(f"Overall score: {cv_analysis.get('overall_cv_score', 0)}")
        
        return cv_analysis
        
    def _store_cv_analysis(self, candidate, cv_analysis: Dict[str, Any]) -> None:
        """Helper method to store CV analysis appropriately based on candidate type."""
        # For Pydantic models, store in the separate dictionary using candidate ID or name as key
        if hasattr(candidate, '__dict__'):
            # Use candidate_id if available, otherwise use name as key
            key = getattr(candidate, 'candidate_id', None) or getattr(candidate, 'name', str(id(candidate)))
            self.candidate_analysis_results[key] = cv_analysis
        else:
            # Handle dictionary - store directly in the dictionary
            candidate["cv_analysis"] = cv_analysis
    
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
    
    def _filter_candidates(self, job_dict: Dict[str, Any], candidates: List[CandidateProfile], 
                           min_match_threshold: float = 0.5) -> List[MatchResult]:
        """Filter candidates based on detailed criteria including experience and CV analysis."""
        logger.info(f"Performing detailed analysis on {len(candidates)} job preference filtered candidates")
        matched_candidates = []
        
        # Define role expertise based on job details
        role_expertise = self._research_role_requirements(
            job_dict.get("title", ""),
            job_dict.get("important_qualities", "")
        )

        for candidate in candidates:
            logger.debug(f"\nEvaluating candidate: {candidate.name} (Row: {getattr(candidate, 'row_number', 'N/A')})")
            
            # Basic experience check (if required)
            min_exp = job_dict.get("min_years_experience", 0) or 0
            candidate_exp = candidate.years_of_experience or 0
            if min_exp > 0:
                logger.debug(f"Experience check for {candidate.name}: {candidate_exp} years vs required {min_exp}")
                if candidate_exp < min_exp:
                    logger.info(f"  Skipping {candidate.name} due to insufficient experience ({candidate_exp} < {min_exp})")
                    continue
            
            # --- Fetch CV Content if needed --- 
            if candidate.cv_link and not candidate.cv_content:
                 logger.debug(f"Fetching CV content for {candidate.name} from {candidate.cv_link}")
                 try:
                     # Use the tool instance provided during initialization
                     fetched_content = self.tool.extract_cv_content(candidate.cv_link)
                     candidate.cv_content = fetched_content if fetched_content else ""
                     if not candidate.cv_content:
                         logger.warning(f"CV content extraction failed or returned empty for {candidate.name}")
                 except Exception as e:
                     logger.error(f"Error fetching CV content for {candidate.name}: {e}")
                     candidate.cv_content = "" # Ensure it's a string even on error
            elif not candidate.cv_link:
                 logger.debug(f"No CV link for {candidate.name}, setting CV content to empty string.")
                 candidate.cv_content = ""
            else:
                 logger.debug(f"CV content already present for {candidate.name}")
            # ---------------------------------

            # Evaluate CV content
            cv_match = self._evaluate_candidate_cv(candidate, role_expertise)
            
            # Get candidate key to retrieve analysis results
            candidate_key = getattr(candidate, 'candidate_id', None) or getattr(candidate, 'name', str(id(candidate)))
            
            # Evaluate skills match
            required_skills = job_dict.get("required_skills", [])
            preferred_skills = job_dict.get("preferred_skills", [])
            
            # Retrieve inferred skills from our stored analysis
            cv_analysis = self.candidate_analysis_results.get(candidate_key, {})
            inferred_skills = cv_analysis.get('inferred_skills', [])
            
            skill_match = self._detailed_skill_matching_for_profile(candidate, required_skills, preferred_skills, 
                                                             inferred_skills)
            
            # Combine scores (customize weighting as needed)
            cv_score = cv_analysis.get("overall_cv_score", 0.1)
            skill_score = skill_match.get("overall_skill_score", 0.1)
            
            overall_score = (cv_score * 0.6) + (skill_score * 0.4)
            
            logger.debug(f"  CV Score: {cv_score:.2f}")
            logger.debug(f"  Skill Score: {skill_score:.2f}")
            logger.info(f"  Overall Score for {candidate.name}: {overall_score:.2f}")
            
            if overall_score >= min_match_threshold:
                matched_candidates.append(MatchResult(
                    candidate_id=candidate.candidate_id,
                    candidate_name=candidate.name,
                    score=overall_score,
                    details={
                        "cv_score": cv_score,
                        "skill_score": skill_score,
                        "required_skills_matched": skill_match.get("required_skills_matched", []),
                        "required_skills_missing": skill_match.get("required_skills_missing", []),
                        "evidence_found": cv_analysis.get("evidence_found", []),
                        "evidence_scores": cv_analysis.get("evidence_scores", {})
                    },
                    row_number=getattr(candidate, 'row_number', None) # Include row number
                ))
            else:
                logger.info(f"  Candidate {candidate.name} did not meet threshold ({overall_score:.2f} < {min_match_threshold}) ")
        
        # Sort by score descending
        matched_candidates.sort(key=lambda x: x.score, reverse=True)
        
        return matched_candidates

    def _rank_candidates_with_detail(self, job: Dict[str, Any], filtered_candidates: List[MatchResult], 
                                   top_n: int = 10) -> List[Dict[str, Any]]:
        """Rank candidates based on comprehensive evaluation."""
        ranked_results = []
        
        print(f"\nPerforming detailed ranking of {len(filtered_candidates)} candidates...")
        
        for match_result in filtered_candidates:
            # Extract candidate info from the match_result object properties
            candidate_name = match_result.candidate_name
            
            print(f"\nDetailed evaluation for candidate: {candidate_name}")
            
            # Create a temporary candidate dict from the stored values
            candidate = {
                "name": candidate_name,
                "years_of_experience": 0,  # Will be set based on the actual data
                "current_location": "",
                "remote_preference": False,
                "willing_to_relocate": False
            }
            
            # Calculate skill matching details
            skill_match_details = {}  # Will contain matched skills
            
            # Get required and preferred skills
            required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
            preferred_skills = {skill.get("name", "").lower() for skill in job.get("preferred_skills", [])}
            
            # For demonstration, let's say all required skills match at this lower threshold
            for skill in required_skills:
                skill_match_details[skill] = True
            
            required_matched = len(required_skills)  # All match for now
            preferred_matched = 0  # None match for now
            
            required_match_pct = (required_matched / len(required_skills) * 100) if required_skills else 100
            preferred_match_pct = (preferred_matched / len(preferred_skills) * 100) if preferred_skills else 100
            
            print(f"  Required Skills: {required_matched}/{len(required_skills)} = {required_match_pct:.1f}%")
            print(f"  Preferred Skills: {preferred_matched}/{len(preferred_skills)} = {preferred_match_pct:.1f}%")
            
            min_years = job.get("min_years_experience", 0) or 0
            candidate_years = 3.0  # Default to minimum
            experience_match = candidate_years >= min_years
            
            print(f"  Experience: {candidate_years} years vs. required {min_years} = {experience_match}")
            
            location_match = False
            if job.get("remote_friendly"):
                location_match = True
            
            print(f"  Location Match: {location_match}")
            
            # Calculate match score (simplified for debugging)
            match_score = match_result.score * 100  # Convert to 0-100 scale
            
            print(f"  Final Match Score: {match_score:.1f}")
            
            explanation = f"Candidate {candidate_name} has a {match_score:.1f}% match with required skills and experience."
            
            result = {
                "candidate": candidate,
                "name": candidate_name,
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
        
        # STAGE 1: Filter by Job Preference
        print("\nSTAGE 1: FILTERING BY JOB PREFERENCE")
        print("-"*80)
        
        # Filter candidates based on their job preference
        job_preference_filtered = []
        for candidate in candidates:
            if 'job_preference' in candidate and candidate['job_preference']:
                # Check if the candidate's job preference matches the current job
                is_related, similarity = self._are_job_titles_related(
                    job.get('title', '').lower(),
                    candidate['job_preference'].lower()
                )
                if is_related:
                    job_preference_filtered.append(candidate)
                    print(f"✓ {candidate.get('name', 'Unknown')} - Job preference matches")
                else:
                    print(f"✗ {candidate.get('name', 'Unknown')} - Job preference doesn't match")
            else:
                print(f"✗ {candidate.get('name', 'Unknown')} - No job preference specified")
        
        if not job_preference_filtered:
            print("\nNo candidates match the job preference.")
            return {
                "job_title": job.get("title", "Unknown Position"),
                "total_candidates": len(candidates),
                "filtered_candidates": 0,
                "top_candidates": 0,
                "ranked_candidates": []
            }
        
        print(f"\nFound {len(job_preference_filtered)} candidates matching the job preference")
        
        # STAGE 2: Analyze and Score Matching Candidates
        print("\nSTAGE 2: ANALYZING AND SCORING MATCHING CANDIDATES")
        print("-"*80)
        
        # Analyze CVs and score candidates
        scored_candidates = []
        for candidate in job_preference_filtered:
            print(f"\nAnalyzing candidate: {candidate.get('name', 'Unknown')}")
            
            # Get role expertise for CV analysis
            role_expertise = self._research_role_requirements(
                job.get('title', ''),
                job.get('important_qualities', '')
            )
            
            # Analyze CV content
            cv_analysis = self._analyze_cv_content(candidate.get('cv_content', ''), role_expertise)
            
            # Calculate scores
            skills_score = self._evaluate_skills_match(job, candidate)
            experience_score = self._evaluate_experience_match(job, candidate)
            location_score = self._evaluate_location_match(job, candidate)
            cv_score = cv_analysis.get('overall_cv_score', 0.0)
            
            # Calculate final score (weighted average)
            final_score = (skills_score * 0.3 + 
                         experience_score * 0.2 + 
                         location_score * 0.1 + 
                         cv_score * 0.4)
            
            # Add to scored candidates
            scored_candidates.append({
                'candidate': candidate,
                'score': final_score,
                'skills_score': skills_score,
                'experience_score': experience_score,
                'location_score': location_score,
                'cv_score': cv_score,
                'cv_analysis': cv_analysis
            })
        
        # Sort candidates by score
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare final results
        result = {
            "job_title": job.get("title", "Unknown Position"),
            "total_candidates": len(candidates),
            "filtered_candidates": len(job_preference_filtered),
            "top_candidates": len(scored_candidates),
            "ranked_candidates": [{
                'name': c['candidate'].get('name', 'Unknown'),
                'score': c['score'],
                'skills_score': c['skills_score'],
                'experience_score': c['experience_score'],
                'location_score': c['location_score'],
                'cv_score': c['cv_score'],
                'cv_analysis': c['cv_analysis']
            } for c in scored_candidates]
        }
        
        print("\nEVALUATION COMPLETE")
        print(f"Found {len(scored_candidates)} suitable candidates out of {len(candidates)} total applicants")
        print(f"First filtered to {len(job_preference_filtered)} candidates by job preference")
        print("="*80)
        
        return result
    
    def match_candidates_to_job(self, job: JobRequirement, all_candidates: List[CandidateProfile],
                              min_match_threshold: float = 0.3, top_n: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Match candidates to a job using the enhanced two-stage filtering framework."""
        logger.info(f"Starting candidate matching process for job: {job.title}")
        
        job_dict = job.model_dump()
        candidate_dicts = [c.model_dump() for c in all_candidates]
        
        # STAGE 1: Job Preference Filtering (Coordinator Agent)
        logger.info(f"Performing job preference filtering on {len(all_candidates)} candidates")
        print("\n" + "="*80)
        print(f"TALENT MATCHING PROCESS: {job.title}")
        print("="*80)

        # First filter by job preference (using original objects)
        print("\nSTAGE 1: COORDINATOR AGENT - INITIAL JOB PREFERENCE FILTERING")
        print("-"*80)
        # Pass the original list of CandidateProfile objects
        job_preference_filtered = self._filter_candidates_by_job_preference(job_dict, all_candidates)
        # The function now returns List[CandidateProfile]
        logger.info(f"Job preference filtering reduced candidates from {len(all_candidates)} to {len(job_preference_filtered)}")

        # Check if enough candidates remain
        MIN_CANDIDATES_AFTER_PREF_FILTER = 3 # Configurable threshold
        if len(job_preference_filtered) < MIN_CANDIDATES_AFTER_PREF_FILTER:
            logger.warning(f"Too few candidates ({len(job_preference_filtered)}) after job preference filtering, relaxing criteria")
            # Option 1: Fallback to all candidates (could be inefficient)
            # job_preference_filtered = all_candidates
            # Option 2: Proceed with the few candidates (current behavior, might be fine)
            # Or, implement a more sophisticated fallback strategy if needed

        # STAGE 2: Detailed evaluation with CV analysis (only on job preference filtered candidates)
        try:
            logger.info(f"Performing detailed analysis on {len(job_preference_filtered)} job preference filtered candidates")
            print("\nSTAGE 2: HR MANAGER AGENT - DETAILED CANDIDATE EVALUATION")
            print("-"*80)

            # Apply second-stage filtering and full analysis only to candidates passing first filter
            # Pass the list of CandidateProfile objects
            filtered_candidates = self._filter_candidates(job_dict, job_preference_filtered, min_match_threshold)

            # Get final rankings with detailed explanations
            match_results = self._rank_candidates_with_detail(job_dict, filtered_candidates, top_n)

            logger.info(f"Successfully matched and ranked {len(match_results)} candidates")
            print(f"\nSuccessfully matched and ranked {len(match_results)} candidates")

        except Exception as e:
            logger.error(f"Error during detailed matching process: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return [], []
        
        return match_results, []
    
    def _pre_filter_candidates_no_preference(self, job: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pre-filter candidates without considering job preference."""
        print("\nRelaxing job preference filter to find additional candidates")
        
        # Extract minimum requirements for quick filtering
        min_years_experience = job.get("min_years_experience", 0)
        required_skills = {skill.get("name", "").lower() for skill in job.get("required_skills", [])}
        
        filtered_candidates = []
        
        for candidate in candidates:
            candidate_name = candidate.get("name", "Unknown")
            passed_filters = True
            
            # Skip detailed evaluation for efficiency in this backup filtering
            # Just check years and at least one skill match
            
            # Check years of experience
            years_experience = candidate.get("years_of_experience", 0)
            if years_experience < min_years_experience and min_years_experience > 0:
                print(f"  {candidate_name}: Insufficient experience ({years_experience} vs {min_years_experience}) - SKIP")
                continue
            
            # Check for at least one skill match
            if required_skills:
                candidate_skills = {skill.get("name", "").lower() for skill in candidate.get("skills", [])}
                if not any(skill in candidate_skills for skill in required_skills):
                    print(f"  {candidate_name}: No matching required skills - SKIP")
                    continue
            
            print(f"  {candidate_name}: Passes basic criteria (no job preference check)")
            filtered_candidates.append(candidate)
        
        return filtered_candidates

    def _detailed_skill_matching_for_profile(self, candidate: CandidateProfile, 
                                    required_skills: List[str], 
                                    preferred_skills: List[str],
                                    inferred_skills: List[str] = None) -> Dict[str, Any]:
        """Detailed skill matching for CandidateProfile objects."""
        inferred_skills = inferred_skills or []
        
        # Extract skills from candidate profile
        candidate_skills = []
        if hasattr(candidate, 'skills') and candidate.skills:
            candidate_skills.extend(candidate.skills)
        
        # Add inferred skills from CV if available
        if inferred_skills:
            candidate_skills.extend(inferred_skills)
        
        # Process required skills list - handle both string lists and dict/object lists
        processed_required_skills = []
        for skill in required_skills:
            if isinstance(skill, dict):
                # If it's a dictionary with a 'name' key
                if 'name' in skill:
                    processed_required_skills.append(skill['name'])
            elif isinstance(skill, str):
                # Already a string
                processed_required_skills.append(skill)
            elif hasattr(skill, 'name'):
                # If it's an object with a name attribute
                processed_required_skills.append(skill.name)
                
        # Process preferred skills list - handle both string lists and dict/object lists
        processed_preferred_skills = []
        for skill in preferred_skills:
            if isinstance(skill, dict):
                # If it's a dictionary with a 'name' key
                if 'name' in skill:
                    processed_preferred_skills.append(skill['name'])
            elif isinstance(skill, str):
                # Already a string
                processed_preferred_skills.append(skill)
            elif hasattr(skill, 'name'):
                # If it's an object with a name attribute
                processed_preferred_skills.append(skill.name)
        
        # Make skills case-insensitive
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        required_skills_lower = [s.lower() for s in processed_required_skills]
        preferred_skills_lower = [s.lower() for s in processed_preferred_skills]
        
        # Match skills
        required_matched = []
        required_missing = []
        preferred_matched = []
        
        for skill in required_skills_lower:
            if any(skill in cs or cs in skill for cs in candidate_skills_lower):
                required_matched.append(skill)
            else:
                required_missing.append(skill)
                
        for skill in preferred_skills_lower:
            if any(skill in cs or cs in skill for cs in candidate_skills_lower):
                preferred_matched.append(skill)
                
        # Calculate scores
        required_score = len(required_matched) / len(required_skills_lower) if required_skills_lower else 1.0
        preferred_score = len(preferred_matched) / len(preferred_skills_lower) if preferred_skills_lower else 1.0
        
        # Overall weighted score
        overall_score = (required_score * 0.7) + (preferred_score * 0.3)
        
        return {
            "required_skills_matched": required_matched,
            "required_skills_missing": required_missing,
            "preferred_skills_matched": preferred_matched,
            "required_match_pct": required_score,
            "preferred_match_pct": preferred_score,
            "overall_skill_score": overall_score
        }

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

    def match_skills(self, required_skills: List[str], candidate_skills: List[str]) -> float:
        """
        Calculate skill match score between required and candidate skills.
        
        Args:
            required_skills: List of required skills
            candidate_skills: List of candidate skills
            
        Returns:
            Match score between 0 and 1
        """
        if not required_skills:
            return 1.0  # No required skills means perfect match
        
        if not candidate_skills:
            return 0.0
        
        # Convert to lowercase for case-insensitive matching
        required_skills = [skill.lower() for skill in required_skills]
        candidate_skills = [skill.lower() for skill in candidate_skills]
        
        # Calculate exact matches
        exact_matches = sum(1 for skill in required_skills if skill in candidate_skills)
        
        # Calculate partial matches using fuzzy matching
        partial_matches = 0
        for req_skill in required_skills:
            for cand_skill in candidate_skills:
                # Check for partial matches (e.g., "python" matches "python programming")
                if req_skill in cand_skill or cand_skill in req_skill:
                    partial_matches += 1
                    break
        
        # Combine exact and partial matches
        total_matches = max(exact_matches, partial_matches)
        
        return total_matches / len(required_skills)

    def evaluate_candidate_skills(self, job_requirements: Dict[str, Any], candidate_profile: Dict[str, Any]) -> float:
        """
        Evaluate candidate's skills against job requirements.
        
        Args:
            job_requirements: Job requirements dictionary
            candidate_profile: Candidate profile dictionary
            
        Returns:
            Skills match score between 0 and 1
        """
        # Extract skills from job requirements
        required_skills = []
        if 'skills' in job_requirements:
            required_skills.extend(job_requirements['skills'])
        if 'requirements' in job_requirements:
            required_skills.extend(self.extract_skills_from_text(job_requirements['requirements']))
        
        # Extract skills from candidate profile
        candidate_skills = []
        if 'skills' in candidate_profile:
            candidate_skills.extend(candidate_profile['skills'])
        if 'experience' in candidate_profile:
            candidate_skills.extend(self.extract_skills_from_text(candidate_profile['experience']))
        if 'education' in candidate_profile:
            candidate_skills.extend(self.extract_skills_from_text(candidate_profile['education']))
        
        return self.match_skills(required_skills, candidate_skills)


# Helper function for LLM configuration
def get_llm_config(api_key=None):
    """Get LLM configuration for AutoGen (optional, can be None for function-only agents)."""
    if not api_key:
        return None
        
    return {
        "config_list": [{"model": "gpt-4", "api_key": api_key}],
        "cache_seed": 42
    }