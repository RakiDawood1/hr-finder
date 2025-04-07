"""
Simple test script to verify CV column access and content extraction.
"""

import os
from dotenv import load_dotenv
from talent_matching_tool import TalentMatchingTool

def test_cv_access():
    # Load environment variables
    load_dotenv()
    
    # Get credentials path from environment
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    
    # Initialize the tool
    tool = TalentMatchingTool(credentials_path)
    
    # Get CV column index from environment (for display purposes)
    cv_column_index = int(os.getenv('CV_COLUMN_INDEX', '0'))
    print(f"Using CV column index: {cv_column_index}")
    
    # Fetch a sample talent with CV (using row 2)
    talent_row = 2
    print(f"\nFetching talent from row {talent_row}...")
    talent_details, cv_content = tool.get_talent_with_cv(talent_row)
    
    # Print talent details
    print("\nTalent details:")
    for key, value in talent_details.items():
        print(f"  {key}: {value}")
    
    # Print CV column name based on the index
    talents_df = tool.get_all_talents()
    if cv_column_index < len(talents_df.columns):
        cv_column_name = talents_df.columns[cv_column_index]
        print(f"\nCV column name based on index {cv_column_index}: '{cv_column_name}'")
    else:
        print(f"\nWarning: CV column index {cv_column_index} is out of range for columns: {list(talents_df.columns)}")
    
    # Print CV content preview
    print("\nCV Content Preview (first 500 chars):")
    print(cv_content[:500] + "..." if len(cv_content) > 500 else cv_content)
    
    # Check if CV content was successfully extracted
    if len(cv_content) > 100:
        print("\n✅ Successfully extracted CV content!")
    elif "Error:" in cv_content:
        print(f"\n❌ Failed to extract CV content: {cv_content}")
    elif "No CV link provided" in cv_content:
        print("\n⚠️ No CV link found for this talent.")
    else:
        print("\n⚠️ CV content seems unusually short, please verify.")

if __name__ == "__main__":
    test_cv_access()