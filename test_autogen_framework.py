"""
Test script for the AutoGen-based Multi-Agent Framework.

This script tests the functionality of the AutoGen-based framework by creating
mock jobs and candidates, then verifying the matching process.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import json

# Import Pydantic models
from pydantic_models import JobRequirement, CandidateProfile, Skill, Education, ExperienceLevel
from autogen_framework import AutoGenTalentMatcher
from talent_matching_tool import TalentMatchingTool
from autogen_matching_engine import AutoGenMatchingEngine

def create_test_job() -> JobRequirement:
    """Create a test job requirement."""
    return JobRequirement(
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

def create_test_candidates() -> List[CandidateProfile]:
    """Create a list of test candidate profiles with varying match qualities."""
    return [
        # Strong match
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
        
        # Moderate match
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
        
        # Weak match
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
        ),
        
        # Poor match
        CandidateProfile(
            candidate_id="CAND-4",
            name="Jordan Lee",
            email="jordan@example.com",
            current_title="Java Developer",
            skills=[
                Skill(name="Java", proficiency=5, years_experience=6),
                Skill(name="Spring", proficiency=4, years_experience=5),
                Skill(name="Hibernate", proficiency=4, years_experience=4),
                Skill(name="SQL", proficiency=3, years_experience=5)
            ],
            years_of_experience=6,
            current_location="Austin",
            willing_to_relocate=False,
            cv_content="Java developer with expertise in Spring Boot and Hibernate. Extensive experience with SQL databases including MySQL and PostgreSQL. No Python experience."
        ),
        
        # Missing required skills but strong in preferred
        CandidateProfile(
            candidate_id="CAND-5",
            name="Casey Wilson",
            email="casey@example.com",
            current_title="DevOps Engineer",
            skills=[
                Skill(name="AWS", proficiency=5, years_experience=6),
                Skill(name="Docker", proficiency=5, years_experience=4),
                Skill(name="Kubernetes", proficiency=5, years_experience=3),
                Skill(name="Python", proficiency=2, years_experience=2)
            ],
            years_of_experience=6,
            current_location="Remote",
            remote_preference=True,
            cv_content="DevOps engineer specializing in cloud infrastructure with AWS. Expert in containerization with Docker and Kubernetes. Basic Python scripting for automation."
        )
    ]

def test_autogen_framework():
    """Test the AutoGen-based framework functionality."""
    print("\n=== Testing AutoGen Talent Matcher ===")
    
    # Create test data
    job = create_test_job()
    candidates = create_test_candidates()
    
    # Initialize the matcher
    matcher = AutoGenTalentMatcher(config_list=None, verbose=True)
    
    # Match candidates to job
    ranked_candidates, _ = matcher.match_candidates_to_job(job, candidates, min_match_threshold=0.3, top_n=3)
    
    # Display results
    print(f"Total candidates: {len(candidates)}")
    print(f"Matched candidates: {len(ranked_candidates)}")
    print("\nTop matched candidates:")
    for i, match in enumerate(ranked_candidates, 1):
        print(f"{i}. {match.get('name', '')} - Match score: {match.get('match_score', 0):.1f}")
        print(f"   Required skills: {match.get('required_skills_matched', '0%')}")
        print(f"   Experience match: {match.get('experience_match', 'No')}, Location match: {match.get('location_match', 'No')}")
        print(f"   Explanation: {match.get('explanation', '')}")
        print()
    
    print("\nAutoGen Talent Matcher test completed.")
    return ranked_candidates

def test_with_real_data():
    """Test the AutoGen framework with real data from Google Sheets (if available)."""
    print("\n=== Testing AutoGen with Real Data ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path or not os.path.exists(credentials_path):
        print("GOOGLE_APPLICATION_CREDENTIALS not set or file not found. Skipping real data test.")
        return
    
    # Initialize components
    try:
        tool = TalentMatchingTool(credentials_path)
        engine = AutoGenMatchingEngine(tool, config_list=None, verbose=True)
        
        # Get a job to test with (we'll use row 2)
        job_row = 2
        result = engine.match_job_to_candidates(job_row, top_n=3)
        
        if result["success"]:
            print(f"Successfully tested with real data!")
            print(f"Job: {result['job_title']} (Row {job_row})")
            print(f"Matched {result['matched_candidates']} candidates from {result['total_candidates']} total")
            print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
            
            # Print top 3 matches if available
            top_matches = result["match_details"][:3]
            if top_matches:
                print("\nTop matches:")
                for i, match in enumerate(top_matches, 1):
                    print(f"{i}. {match.get('name', '')} - Score: {match.get('match_score', 0):.1f}")
        else:
            print(f"Error testing with real data: {result['error']}")
    
    except Exception as e:
        print(f"Error during real data test: {str(e)}")
    
    print("\nReal data test completed.")

def compare_with_custom_framework():
    """Compare AutoGen results with the custom framework."""
    print("\n=== Comparing AutoGen with Custom Framework ===")
    
    try:
        # Import the custom framework
        
        from agent_framework import AgentFramework
        
        # Create test data
        job = create_test_job()
        candidates = create_test_candidates()
        
        # Initialize both frameworks
        autogen_matcher = AutoGenTalentMatcher(config_list=None, verbose=False)
        custom_framework = AgentFramework()
        
        # Match using both frameworks
        print("Running AutoGen matcher...")
        autogen_results, _ = autogen_matcher.match_candidates_to_job(job, candidates, min_match_threshold=0.3, top_n=5)
        
        print("Running custom framework...")
        custom_results, _ = custom_framework.get_top_candidates_for_job(job, candidates, min_match_threshold=0.3, top_n=5)
        
        # Compare results
        print("\nTop candidates from AutoGen:")
        for i, match in enumerate(autogen_results[:3], 1):
            print(f"{i}. {match.get('name', '')} - Score: {match.get('match_score', 0):.1f}")
        
        print("\nTop candidates from Custom Framework:")
        for i, match in enumerate(custom_results[:3], 1):
            print(f"{i}. {match.candidate.name} - Score: {match.match_score:.1f}")
        
        # Check for agreement
        autogen_names = [match.get('name', '') for match in autogen_results]
        custom_names = [match.candidate.name for match in custom_results]
        
        # Calculate similarity in rankings
        common_names = set(autogen_names).intersection(set(custom_names))
        similarity_percentage = (len(common_names) / len(autogen_names)) * 100 if autogen_names else 0
        
        print(f"\nSimilarity between frameworks: {similarity_percentage:.1f}%")
        print(f"Common candidates in top results: {len(common_names)} of {len(autogen_names)}")
        print(f"Common candidates: {', '.join(common_names)}")
        
    except ImportError:
        print("Custom framework not found. Skipping comparison.")
    
    print("\nComparison completed.")

def main():
    """Run all tests for the AutoGen framework."""
    print("==== AutoGen Multi-Agent Framework Testing ====")
    
    # Test the AutoGen framework
    ranked_candidates = test_autogen_framework()
    
    # Test with real data if available
    test_with_real_data()
    
    # Compare with custom framework if available
    compare_with_custom_framework()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()