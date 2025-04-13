"""
Pydantic integration for the Talent Matching Tool.

This module provides functions to convert raw data from Google Sheets and CV content
into standardized Pydantic models for consistent data processing and validation.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pydantic_models import JobRequirement, CandidateProfile, Skill, Education, ExperienceLevel


def parse_job_to_model(job_data: Dict[str, Any]) -> JobRequirement:
    """
    Convert raw job data from Google Sheets into a validated JobRequirement model.
    
    Args:
        job_data: Dictionary containing raw job data from Google Sheets
        
    Returns:
        JobRequirement: A validated job requirement model
    """
    # Extract basic job details
    job_model = JobRequirement(
        job_id=job_data.get('JobID', job_data.get('Job ID', None)),
        title=job_data.get('Title', job_data.get('JobTitle', job_data.get('Job Title', ''))),
        department=job_data.get('Department', None),
        location=job_data.get('Location', None),
        description=job_data.get('Description', job_data.get('JobDescription', job_data.get('Job Description', None))),
        remote_friendly=_parse_boolean(job_data.get('RemoteFriendly', job_data.get('Remote Friendly', 'No')))
    )
    
    # Try to parse experience level
    experience_text = job_data.get('ExperienceLevel', job_data.get('Experience Level', ''))
    try:
        job_model.experience_level = _parse_experience_level(experience_text)
    except ValueError:
        # If we can't map directly, leave as None
        pass
    
    # Parse years of experience
    years_exp_text = job_data.get('YearsExperience', job_data.get('Years Experience', 
                                  job_data.get('MinYearsExperience', job_data.get('Min Years Experience', '0'))))
    job_model.min_years_experience = _extract_number(years_exp_text)
    
    # Parse required skills
    required_skills_text = job_data.get('RequiredSkills', job_data.get('Required Skills', ''))
    job_model.required_skills = _parse_skills(required_skills_text)
    
    # Parse preferred skills
    preferred_skills_text = job_data.get('PreferredSkills', job_data.get('Preferred Skills', ''))
    job_model.preferred_skills = _parse_skills(preferred_skills_text)
    
    # Parse qualifications and responsibilities from description if not directly provided
    if job_model.description:
        if not job_data.get('Qualifications'):
            job_model.qualifications = _extract_qualifications(job_model.description)
        
        if not job_data.get('Responsibilities'):
            job_model.responsibilities = _extract_responsibilities(job_model.description)
    
    # Parse salary range if available
    salary_text = job_data.get('SalaryRange', job_data.get('Salary Range', ''))
    if salary_text:
        salary_range = _parse_salary_range(salary_text)
        if salary_range:
            job_model.salary_range = salary_range
    
    return job_model


def parse_candidate_to_model(candidate_data: Dict[str, Any], cv_content: Optional[str] = None, row_number: Optional[int] = None) -> CandidateProfile:
    """
    Convert raw candidate data from Google Sheets into a validated CandidateProfile model.
    
    Args:
        candidate_data: Dictionary containing raw candidate data from Google Sheets
        cv_content: Optional text content extracted from the candidate's CV
        row_number: Optional row number for reference
        
    Returns:
        CandidateProfile: A validated candidate profile model
    """
    # Use CV content from arguments if provided, otherwise use from data
    if cv_content is not None:
        candidate_data["cv_content"] = cv_content
    
    # Extract name or use default if not present
    full_name = (candidate_data.get('name', candidate_data.get('Name', candidate_data.get('FullName', 
                                    candidate_data.get('Full Name', '')))))
    
    # Ensure name is not empty
    if not full_name:
        full_name = f"Candidate-{row_number or candidate_data.get('row_number', 'Unknown')}"
    
    print(f"Processing candidate: {full_name}")
    print(f"CV content length: {len(cv_content) if cv_content else 0}")
    
    if cv_content:
        print("First 100 chars of CV content:")
        print("-" * 30)
        print(cv_content[:100])
        print("-" * 30)
    
    # Process skills
    skills_text = candidate_data.get('Skills', candidate_data.get('skills', ''))
    skills = []
    
    # Handle different formats of skills data
    if isinstance(skills_text, list):
        # If skills is already a list (might be a list of skill objects or strings)
        skills = skills_text
    elif skills_text:
        # If skills is a string, parse it
        try:
            skills = _parse_skills(skills_text)
        except Exception as e:
            print(f"Warning: Error parsing skills text '{skills_text}': {e}")
            skills = []  # Fallback to empty list
    
    # Build basic model, including row_number
    candidate_model = CandidateProfile(
        candidate_id=candidate_data.get('CandidateID', candidate_data.get('Candidate ID', None)),
        name=full_name,
        email=candidate_data.get('Email', None),
        phone=candidate_data.get('Phone', candidate_data.get('PhoneNumber', candidate_data.get('Phone Number', None))),
        current_title=candidate_data.get('CurrentTitle', candidate_data.get('Current Title', 
                                        candidate_data.get('JobTitle', candidate_data.get('Job Title', None)))),
        current_location=candidate_data.get('Location', candidate_data.get('CurrentLocation', 
                                           candidate_data.get('Current Location', None))),
        willing_to_relocate=_parse_boolean(candidate_data.get('WillingToRelocate', 
                                           candidate_data.get('Willing To Relocate', 'No'))),
        remote_preference=_parse_boolean(candidate_data.get('RemotePreference', 
                                         candidate_data.get('Remote Preference', 'No'))),
        cv_content=candidate_data.get('cv_content', ''),
        cv_link=candidate_data.get('CV', candidate_data.get('Resume', 
                                   candidate_data.get('CVLink', candidate_data.get('CV Link', None)))),
        jobs_applying_for=candidate_data.get('jobs_applying_for', candidate_data.get('position_preference', None)),
        skills=skills,
        row_number=row_number
    )
    
    # Verify CV content was properly set
    print(f"Model CV content length: {len(candidate_model.cv_content) if candidate_model.cv_content else 0}")
    
    # Parse years of experience
    years_exp_text = candidate_data.get('years_of_experience', candidate_data.get('YearsExperience', candidate_data.get('Years Experience', 
                                        candidate_data.get('TotalExperience', candidate_data.get('Total Experience', '0')))))
    
    # Convert years of experience to float, handling different input types
    try:
        if isinstance(years_exp_text, (int, float)):
            candidate_model.years_of_experience = float(years_exp_text)
        else:
            candidate_model.years_of_experience = _extract_number(str(years_exp_text))
    except (ValueError, TypeError):
        candidate_model.years_of_experience = 0.0
        
    print(f"Years of experience: {candidate_model.years_of_experience}")
    
    # If CV content is available, try to extract more information
    # REMOVED: CV enhancement logic is moved to the analysis stage
    # if cv_content:
    #     print("Enhancing model with CV content...")
    #     candidate_model = _enhance_candidate_with_cv_content(candidate_model, cv_content)
    #     print(f"Enhanced CV content length: {len(candidate_model.cv_content) if candidate_model.cv_content else 0}")
    
    return candidate_model


def _parse_skills(skills_text: str) -> List[Skill]:
    """
    Parse a comma or semicolon separated list of skills into Skill objects.
    
    Args:
        skills_text: String containing skills (e.g., "Python, Java, SQL")
        
    Returns:
        List of Skill objects
    """
    if not skills_text:
        return []
    
    # Split by comma or semicolon
    skill_items = re.split(r'[,;]', skills_text)
    skills = []
    
    for item in skill_items:
        item = item.strip()
        if not item:
            continue
        
        # Check if there's a proficiency indicator like "Python (5 years)" or "Python (Expert)"
        match = re.match(r'^(.*?)\s*(?:\((\d+)(?:\s*years?)?\)|\((\w+)\))$', item)
        
        if match:
            skill_name = match.group(1).strip()
            years = match.group(2)
            level_text = match.group(3)
            
            if years:
                # Years of experience specified
                skills.append(Skill(
                    name=skill_name,
                    years_experience=float(years)
                ))
            elif level_text:
                # Proficiency level specified
                proficiency = _map_text_to_proficiency(level_text)
                skills.append(Skill(
                    name=skill_name,
                    proficiency=proficiency
                ))
            else:
                skills.append(Skill(name=skill_name))
        else:
            skills.append(Skill(name=item))
    
    return skills


def _map_text_to_proficiency(level_text: str) -> int:
    """
    Map textual proficiency levels to numeric values (1-5).
    
    Args:
        level_text: String description of proficiency
        
    Returns:
        Integer proficiency level from 1-5
    """
    level_text = level_text.lower()
    
    if level_text in ['beginner', 'basic', 'novice']:
        return 1
    elif level_text in ['elementary', 'limited working']:
        return 2
    elif level_text in ['intermediate', 'working', 'competent']:
        return 3
    elif level_text in ['advanced', 'highly competent', 'proficient']:
        return 4
    elif level_text in ['expert', 'native', 'fluent', 'master']:
        return 5
    else:
        # Default to middle value if unknown
        return 3


def _parse_experience_level(level_text: str) -> ExperienceLevel:
    """
    Parse experience level text into standardized enum.
    
    Args:
        level_text: String description of experience level
        
    Returns:
        ExperienceLevel enum value
    Raises:
        ValueError: If the level_text cannot be mapped to an ExperienceLevel
    """
    level_text = level_text.lower().strip()
    
    mapping = {
        'entry': ExperienceLevel.ENTRY,
        'entry level': ExperienceLevel.ENTRY,
        'entry-level': ExperienceLevel.ENTRY,
        'junior': ExperienceLevel.JUNIOR,
        'mid': ExperienceLevel.MID,
        'mid level': ExperienceLevel.MID,
        'mid-level': ExperienceLevel.MID,
        'intermediate': ExperienceLevel.MID,
        'senior': ExperienceLevel.SENIOR,
        'senior level': ExperienceLevel.SENIOR,
        'senior-level': ExperienceLevel.SENIOR,
        'lead': ExperienceLevel.LEAD,
        'team lead': ExperienceLevel.LEAD,
        'manager': ExperienceLevel.MANAGER,
        'director': ExperienceLevel.DIRECTOR,
        'executive': ExperienceLevel.EXECUTIVE,
        'c-level': ExperienceLevel.EXECUTIVE,
        'c level': ExperienceLevel.EXECUTIVE
    }
    
    if level_text in mapping:
        return mapping[level_text]
    
    # Try more flexible matching
    for key, value in mapping.items():
        if key in level_text:
            return value
    
    raise ValueError(f"Could not map '{level_text}' to an experience level")


def _parse_boolean(text: str) -> bool:
    """
    Parse various text representations of boolean values.
    
    Args:
        text: String representation of a boolean value
        
    Returns:
        Boolean value
    """
    if not text:
        return False
        
    text = str(text).lower().strip()
    return text in ['yes', 'true', 'y', '1', 't']


def _extract_number(text: str) -> Optional[float]:
    """
    Extract a numeric value from text.
    
    Args:
        text: String potentially containing a number
        
    Returns:
        Float value if found, None otherwise
    """
    if not text:
        return None
        
    # Try direct conversion first
    try:
        return float(text)
    except ValueError:
        pass
    
    # Look for patterns like "5 years" or "3+"
    match = re.search(r'(\d+\.?\d*)\+?', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None


def _parse_salary_range(salary_text: str) -> Dict[str, float]:
    """
    Parse salary range text into min/max values.
    
    Args:
        salary_text: String describing a salary range (e.g., "$50,000 - $70,000")
        
    Returns:
        Dictionary with min and max values if successful, empty dict otherwise
    """
    if not salary_text:
        return {}
    
    # Remove currency symbols and commas
    clean_text = re.sub(r'[$,]', '', salary_text)
    
    # Look for patterns like "50000-70000" or "50000 to 70000"
    match = re.search(r'(\d+\.?\d*)\s*[-–—to]*\s*(\d+\.?\d*)', clean_text)
    if match:
        try:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            return {"min": min_val, "max": max_val}
        except ValueError:
            pass
    
    # Look for single value with "+" indicating minimum
    match = re.search(r'(\d+\.?\d*)\s*\+', clean_text)
    if match:
        try:
            min_val = float(match.group(1))
            return {"min": min_val}
        except ValueError:
            pass
    
    return {}


def _extract_qualifications(description: str) -> List[str]:
    """
    Extract qualification bullet points from job description.
    
    Args:
        description: Full job description text
        
    Returns:
        List of qualification statements
    """
    qualifications = []
    
    # Look for qualifications section
    qual_section_match = re.search(
        r'(?:qualifications|requirements|you should have|what you\'ll need)(?:[:\s]*)(.+?)(?:(?:responsibilities|about the company|about us|what you\'ll do)|$)',
        description.lower(),
        re.DOTALL
    )
    
    if qual_section_match:
        qual_text = qual_section_match.group(1).strip()
        
        # Extract bullet points - may be marked with *, -, • or numbers
        bullet_points = re.findall(r'(?:^|\n)(?:[•\*\-]|\d+\.)\s*(.+?)(?=(?:\n[•\*\-]|\n\d+\.|\Z))', qual_text)
        
        if bullet_points:
            qualifications = [point.strip() for point in bullet_points if point.strip()]
    
    return qualifications


def _extract_responsibilities(description: str) -> List[str]:
    """
    Extract responsibility bullet points from job description.
    
    Args:
        description: Full job description text
        
    Returns:
        List of responsibility statements
    """
    responsibilities = []
    
    # Look for responsibilities section
    resp_section_match = re.search(
        r'(?:responsibilities|what you\'ll do|duties|the role)(?:[:\s]*)(.+?)(?:(?:qualifications|requirements|you should have|what you\'ll need|about the company|about us)|$)',
        description.lower(),
        re.DOTALL
    )
    
    if resp_section_match:
        resp_text = resp_section_match.group(1).strip()
        
        # Extract bullet points - may be marked with *, -, • or numbers
        bullet_points = re.findall(r'(?:^|\n)(?:[•\*\-]|\d+\.)\s*(.+?)(?=(?:\n[•\*\-]|\n\d+\.|\Z))', resp_text)
        
        if bullet_points:
            responsibilities = [point.strip() for point in bullet_points if point.strip()]
    
    return responsibilities