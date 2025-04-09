"""
Simple test to verify the Pydantic integration is working.
"""

import os
from dotenv import load_dotenv
from talent_matching_tool import TalentMatchingTool  # Note: use your actual class name here
from pydantic_models import JobRequirement, CandidateProfile

def verify_pydantic_integration():
    """
    Basic verification of Pydantic model integration.
    """
    print("===== Verifying Pydantic Integration =====")
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials path
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    
    try:
        # Verify we can import the models
        print("✅ Successfully imported Pydantic models")
        
        # Initialize the tool
        tool = TalentMatchingTool(credentials_path)  # Use your actual class name
        print("✅ Successfully initialized tool")
        
        # Try to load and parse a job
        job_row = 2  # Change this if needed
        print(f"\nAttempting to load job from row {job_row}...")
        
        job_model = tool.get_job_model(job_row)  # This method should exist in your enhanced implementation
        
        if job_model:
            print(f"✅ Successfully loaded and parsed job model: {job_model.title}")
            print(f"  Required skills: {', '.join([skill.name for skill in job_model.required_skills]) if job_model.required_skills else 'None'}")
        else:
            print("❌ Failed to load job model")
        
        print("\nVerification complete!")
        
    except Exception as e:
        print(f"❌ Error during verification: {type(e).__name__}: {e}")

if __name__ == "__main__":
    verify_pydantic_integration()