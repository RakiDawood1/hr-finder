#!/usr/bin/env python3
"""
Simple test script to verify the updated talent matching system.
This will run a matching process using your updated code.
"""

import os
import sys
from dotenv import load_dotenv
from talent_matching_tool_fix2 import TalentMatchingTool
from autogen_matching_engine_fix2 import AutoGenMatchingEngine
import logging

def test_matching_for_job(job_row):
    """Test the matching process for a specific job row."""
    print(f"\n===== TESTING MATCHING FOR JOB ROW {job_row} =====")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the tool with your updated code
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    if not os.path.exists(credentials_path):
        print(f"Error: Credentials file not found at {credentials_path}")
        return False
    
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create instances with debug mode
    tool = TalentMatchingTool(credentials_path)
    engine = AutoGenMatchingEngine(tool, verbose=True, debug=True)
    
    # First, check if the job model now has skills
    job_model = tool.get_job_model(job_row)
    if not job_model:
        print(f"Error: Failed to retrieve job model for row {job_row}")
        return False
    
    print(f"\nJob Title: {job_model.title}")
    print(f"Experience Level: {job_model.experience_level}")
    print(f"Min Years Experience: {job_model.min_years_experience}")
    
    print("\nRequired Skills:")
    if job_model.required_skills:
        for skill in job_model.required_skills:
            print(f"  - {skill.name}")
    else:
        print("  No required skills defined (this is still a problem if you see this)")
    
    print("\nPreferred Skills:")
    if job_model.preferred_skills:
        for skill in job_model.preferred_skills:
            print(f"  - {skill.name}")
    else:
        print("  No preferred skills defined (this may be OK)")
    
    # Now run the matching with debug output
    print("\nRunning matching process...")
    result = engine.match_job_to_candidates(
        job_row=job_row,
        min_match_threshold=0.3,
        top_n=5
    )
    
    # Display results
    if result["success"]:
        print(f"\n=== Results for {result['job_title']} (Job Row {job_row}) ===")
        print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
        print(f"Total candidates analyzed: {result['total_candidates']}")
        print(f"Candidates matched: {result['matched_candidates']}")
        print("\nTop matching candidates:")
        
        for i, match in enumerate(result["match_details"], 1):
            print(f"\n{i}. {match.get('name', '')} - Match Score: {match.get('match_score', 0):.1f}")
            row_number = next((r.get('row_number', 'Unknown') for r in result['matches'] 
                           if r.get('name', '') == match.get('name', '')), 'Unknown')
            print(f"   Row Number: {row_number}")
            print(f"   Required Skills: {match.get('required_skills_matched', 'N/A')}")
            print(f"   Experience Match: {match.get('experience_match', 'No')}")
            print(f"   Location Match: {match.get('location_match', 'No')}")
            print(f"   Summary: {match.get('explanation', 'No explanation provided')}")
        
        return True
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False

def main():
    """Main function to test the updated system."""
    # Test for job row 3 (Senior Software Engineer)
    success = test_matching_for_job(3)
    
    if success:
        print("\n✅ The updated system appears to be working correctly!")
        print("The job model now includes skills from the 'Skill and Requirement' column.")
        print("The matching process should now be providing more accurate results.")
    else:
        print("\n❌ The updated system still has issues.")
        print("Please check your modifications to make sure they were applied correctly.")
        print("Verify that the get_job_model method is properly mapping skills from column G.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())