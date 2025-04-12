"""
Pydantic models for the Talent Matching Tool.

This module defines structured data models for job requirements and candidate profiles,
ensuring consistent data validation and standardization throughout the application.
"""

from typing import List, Dict, Optional, Set, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import date


class ExperienceLevel(str, Enum):
    """Standardized experience levels for jobs and candidates."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class Skill(BaseModel):
    """Model representing a professional skill with proficiency level."""
    name: str
    proficiency: Optional[int] = Field(None, ge=1, le=5)  # 1-5 scale if provided
    years_experience: Optional[float] = None
    
    @validator('name')
    def normalize_skill_name(cls, v):
        """Normalize skill names to lowercase for better matching."""
        return v.lower().strip() if v else v


class Education(BaseModel):
    """Model for educational qualifications."""
    degree: str
    field_of_study: str
    institution: str
    graduation_date: Optional[Union[date, str]] = None
    gpa: Optional[float] = None
    
    @validator('field_of_study', 'degree')
    def normalize_education_fields(cls, v):
        """Normalize education fields to lowercase for better matching."""
        return v.lower().strip() if v else v


class JobRequirement(BaseModel):
    """Model for job requirements."""
    job_id: Optional[str] = None
    title: str
    department: Optional[str] = None
    experience_level: Optional[ExperienceLevel] = None
    min_years_experience: Optional[float] = Field(None, ge=0)
    required_skills: List[Skill] = []
    preferred_skills: List[Skill] = []
    required_education: Optional[Education] = None
    preferred_education: List[Education] = []
    location: Optional[str] = None
    remote_friendly: Optional[bool] = False
    description: Optional[str] = None
    responsibilities: List[str] = []
    qualifications: List[str] = []
    salary_range: Optional[Dict[str, float]] = None
    important_qualities: Optional[str] = None  # Added for Column I content
    
    class Config:
        validate_assignment = True
    
    def get_all_skills(self) -> Set[str]:
        """Get all unique skills (both required and preferred) for this job."""
        all_skills = set()
        all_skills.update([skill.name for skill in self.required_skills])
        all_skills.update([skill.name for skill in self.preferred_skills])
        return all_skills


class CVAnalysis(BaseModel):
    """Model for CV content analysis results."""
    evidence_found: List[str] = []
    evidence_scores: Dict[str, float] = {}
    project_evaluation: List[Dict[str, Any]] = []
    leadership_indicators: List[str] = []
    overall_cv_score: float = 0.0


class CandidateProfile(BaseModel):
    """Model for candidate profiles."""
    candidate_id: Optional[str] = None
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    current_title: Optional[str] = None
    skills: List[Skill] = []
    experience: List[Dict[str, Union[str, float]]] = []  # List of work experiences
    education: List[Education] = []
    certifications: List[str] = []
    languages: List[Dict[str, Union[str, int]]] = []  # [{"name": "English", "proficiency": 5}]
    current_location: Optional[str] = None
    willing_to_relocate: Optional[bool] = False
    remote_preference: Optional[bool] = False
    years_of_experience: Optional[float] = Field(None, ge=0)
    cv_content: Optional[str] = None
    cv_link: Optional[str] = None
    jobs_applying_for: Optional[str] = None
    
    class Config:
        validate_assignment = True
    
    def get_all_skill_names(self) -> Set[str]:
        """
        Get all unique skill names for this candidate.
        
        Returns:
            A set of all unique skill names.
        """
        return {skill.name for skill in self.skills}


class MatchResult(BaseModel):
    """Model for representing a job-candidate match result."""
    job: JobRequirement
    candidate: CandidateProfile
    match_score: float = Field(..., ge=0, le=100)
    skill_match_details: Dict[str, bool] = {}
    required_skill_match_percentage: float = Field(..., ge=0, le=100)
    preferred_skill_match_percentage: float = Field(..., ge=0, le=100)
    education_match: bool = False
    experience_match: bool = False
    location_match: bool = False
    cv_analysis: Optional[CVAnalysis] = None  # Added for enhanced CV analysis
    match_explanation: Optional[str] = None
    
    class Config:
        validate_assignment = True