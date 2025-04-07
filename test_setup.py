"""
Test script to verify the environment setup and Google API access.
"""

import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import sys

# Print current working directory
print(f"Current directory: {os.getcwd()}")

# Check if .env file exists
env_path = os.path.join(os.getcwd(), '.env')
print(f".env file exists: {os.path.exists(env_path)}")

# Try to load env file with verbose output
load_dotenv(verbose=True)

# Print all environment variables (be careful if sensitive info)
print("Environment variables:")
for key, value in os.environ.items():
    if 'SHEET_ID' in key or key == 'GOOGLE_APPLICATION_CREDENTIALS':
        print(f"  {key}: {value}")

def test_environment():
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    jobs_sheet_id = os.getenv('JOBS_SHEET_ID')
    talents_sheet_id = os.getenv('TALENTS_SHEET_ID')
    
    print("===== Environment Variables =====")
    print(f"Credentials path: {credentials_path}")
    print(f"Jobs Sheet ID: {jobs_sheet_id}")
    print(f"Talents Sheet ID: {talents_sheet_id}")
    
    # Verify if credentials file exists
    if not os.path.exists(credentials_path):
        print(f"ERROR: Credentials file not found at {credentials_path}")
        return False
    
    print("\n===== Testing Google API Access =====")
    try:
        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
        )
        
        print("✓ Credentials loaded successfully")
        
        # Test Drive API
        drive_service = build('drive', 'v3', credentials=credentials)
        drive_files = drive_service.files().list(pageSize=1).execute()
        print("✓ Drive API access successful")
        
        # Test Sheets API - Jobs Sheet
        sheets_service = build('sheets', 'v4', credentials=credentials)
        result = sheets_service.spreadsheets().get(
            spreadsheetId=jobs_sheet_id
        ).execute()
        print(f"✓ Jobs Sheet access successful: {result.get('properties', {}).get('title')}")
        
        # Test Sheets API - Talents Sheet
        result = sheets_service.spreadsheets().get(
            spreadsheetId=talents_sheet_id
        ).execute()
        print(f"✓ Talents Sheet access successful: {result.get('properties', {}).get('title')}")
        
        print("\n===== SETUP SUCCESSFUL =====")
        print("Your environment is configured correctly!")
        return True
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_environment()