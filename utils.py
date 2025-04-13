"""
Utility functions for the Talent Matching Tool.

This module provides shared functionality used across different components
of the application to reduce code duplication.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("talent_matching_utils")

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing extra whitespace."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using multiple patterns."""
    if not text:
        return []
        
    skills = set()
    text_lower = text.lower()
    
    # Define skill patterns
    skill_patterns = [
        # Programming Languages & Core Libraries
        r'\b(python|r|sql|java|scala|c\+\+|julia)\b',
        r'\b(pandas|numpy|scipy|scikit-learn|sklearn|matplotlib|seaborn|plotly)\b',
        r'\b(tensorflow|keras|pytorch|theano|caffe|mxnet|jax)\b',
        r'\b(nltk|spacy|gensim|transformers)\b',
        r'\b(opencv|pillow)\b',
        
        # Databases & Data Warehousing
        r'\b(sql|mysql|postgresql|postgres|sqlite|sql server|tsql|pl/sql|oracle|mongodb|cassandra|redis|neo4j)\b',
        
        # Machine Learning & AI
        r'\b(machine learning|ml|deep learning|dl|artificial intelligence|ai)\b',
        r'\b(natural language processing|nlp|computer vision|cv)\b',
        
        # Data Science & Analytics
        r'\b(data science|data analysis|statistics|analytics|bi|business intelligence)\b',
        r'\b(etl|data pipeline|data warehouse|data modeling|data engineering)\b',
        
        # Finance & Domain Knowledge
        r'\b(finance|financial|banking|fintech|trading|investment|risk management)\b',
        
        # Other technical skills
        r'\b(git|docker|kubernetes|cloud|aws|azure|gcp)\b'
    ]
    
    # Extract skills
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            skills.add(match.strip())
    
    return list(skills)

def extract_years_experience(text: str) -> Optional[float]:
    """Extract years of experience from text."""
    if not text:
        return None
        
    # Handle numeric values directly
    if isinstance(text, (int, float)):
        return float(text)
    
    # Convert to string and look for patterns
    text = str(text).lower()
    
    # Look for "X years" pattern
    years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text)
    if years_match:
        return float(years_match.group(1))
    
    # Look for just numbers
    numbers_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if numbers_match:
        return float(numbers_match.group(1))
    
    return None

def extract_cv_sections(cv_content: str) -> Dict[str, str]:
    """Extract different sections from CV content."""
    if not cv_content:
        return {"experience": ""}
    
    cv_lower = cv_content.lower()
    sections = {}
    
    # Define section headers
    section_patterns = {
        "education": [r'education[\s]*:?', r'academic[\s]*background[\s]*:?'],
        "experience": [r'experience[\s]*:?', r'work[\s]*history[\s]*:?', r'employment[\s]*history[\s]*:?'],
        "skills": [r'skills[\s]*:?', r'technical[\s]*skills[\s]*:?', r'core[\s]*competencies[\s]*:?'],
        "projects": [r'projects[\s]*:?', r'key[\s]*projects[\s]*:?'],
        "summary": [r'summary[\s]*:?', r'profile[\s]*:?', r'professional\s+summary[\s]*:?']
    }
    
    # Find section positions
    section_positions = []
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, cv_lower):
                section_positions.append((match.start(), section_name))
    
    # Sort by position
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