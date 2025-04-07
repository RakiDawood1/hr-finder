"""
Talent Matching Tool - Phase 1: Setup & Data Access with Pydantic Integration
This script handles Google Sheets and Drive integration to access job requirements and candidate profiles,
with added Pydantic data validation.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import re
import io
import tempfile
from dotenv import load_dotenv

# For document parsing
import PyPDF2
import docx2txt

# Import Pydantic models
from pydantic_models import JobRequirement, CandidateProfile, MatchResult
from pydantic_integration import parse_job_to_model, parse_candidate_to_model

# Load environment variables
load_dotenv()

class TalentMatchingTool:
    """Enhanced main class for the Talent Matching Tool with Pydantic integration."""
    
    def __init__(self, credentials_path: str):
        """
        Initialize the tool with Google API credentials.
        
        Args:
            credentials_path: Path to the service account credentials JSON file
        """
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
        )
        
        # Initialize Google Sheets API client
        self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
        
        # Initialize Google Drive API client
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        
        # Store spreadsheet IDs
        self.jobs_sheet_id = os.getenv('JOBS_SHEET_ID')
        self.talents_sheet_id = os.getenv('TALENTS_SHEET_ID')
        
        # Define sheet names/ranges (can be customized)
        self.jobs_range = os.getenv('JOBS_RANGE', 'Sheet1!A:Z')
        self.talents_range = os.getenv('TALENTS_RANGE', 'Sheet1!A:Z')
        
        # Column index for CV links in talents sheet
        self.cv_column_index = int(os.getenv('CV_COLUMN_INDEX', '0'))
        
    def get_job_details(self, row_number: int) -> Dict[str, Any]:
        """
        Retrieve details for a specific job from the Jobs Sheet.
        
        Args:
            row_number: The row number of the job in the Jobs Sheet
            
        Returns:
            A dictionary containing the job details
        """
        try:
            # Get all data to find headers
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.jobs_sheet_id,
                range=self.jobs_range
            ).execute()
            
            values = result.get('values', [])
            
            if not values or len(values) < row_number:
                raise ValueError(f"No data found or row {row_number} does not exist")
            
            # Extract headers and job data
            headers = values[0]
            job_data = values[row_number - 1] if row_number > 0 else values[0]
            
            # Create a dictionary of job details
            job_dict = {}
            for i, header in enumerate(headers):
                if i < len(job_data):
                    job_dict[header] = job_data[i]
                else:
                    job_dict[header] = ""
                    
            return job_dict
            
        except HttpError as error:
            print(f"An error occurred: {error}")
            return {}
    
    def get_job_model(self, row_number: int) -> Optional[JobRequirement]:
        """
        Retrieve and validate job details as a Pydantic model.
        
        Args:
            row_number: The row number of the job in the Jobs Sheet
            
        Returns:
            A validated JobRequirement model or None if retrieval fails
        """
        job_dict = self.get_job_details(row_number)
        if not job_dict:
            return None
            
        try:
            return parse_job_to_model(job_dict)
        except Exception as e:
            print(f"Error parsing job to model: {e}")
            return None
    
    def get_all_talents(self) -> pd.DataFrame:
        """
        Retrieve all talent profiles from the Talents Sheet.
        
        Returns:
            A pandas DataFrame containing all talent data
        """
        try:
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.talents_sheet_id,
                range=self.talents_range
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                print("No talent data found")
                return pd.DataFrame()
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Add row numbers for reference (starting from 2 because row 1 is headers)
            df.insert(0, 'row_number', range(2, len(df) + 2))
            
            return df
            
        except HttpError as error:
            print(f"An error occurred: {error}")
            return pd.DataFrame()
    
    def extract_file_id_from_drive_link(self, drive_link: str) -> str:
        """
        Extract the file ID from various formats of Google Drive links.
        
        Args:
            drive_link: Google Drive link to a document
            
        Returns:
            The file ID extracted from the link
        """
        if not drive_link:
            return ""
            
        # Pattern for various Google Drive link formats
        patterns = [
            r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",  # /file/d/ format
            r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",  # open?id= format
            r"https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)",  # Google Docs
            r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)",  # Google Sheets
            r"https://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",  # Google Slides
            r"https://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",  # Folders
            r"id=([a-zA-Z0-9_-]+)",  # Generic id= parameter
            r"https://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)"  # Direct download link
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_link)
            if match:
                return match.group(1)
                
        # If no patterns match but link contains a long alphanumeric string, attempt to extract it
        alphanumeric_pattern = r"[a-zA-Z0-9_-]{25,}"
        match = re.search(alphanumeric_pattern, drive_link)
        if match:
            return match.group(0)
            
        return ""
    
    def download_and_extract_text_from_drive_file(self, file_id: str) -> str:
        """
        Download a file from Google Drive and extract its text content.
        
        Args:
            file_id: The ID of the file in Google Drive
            
        Returns:
            The extracted text content
        """
        if not file_id:
            return ""
            
        try:
            # Get file metadata to determine file type
            file_metadata = self.drive_service.files().get(fileId=file_id, fields="name,mimeType").execute()
            file_name = file_metadata.get("name", "")
            mime_type = file_metadata.get("mimeType", "")
            
            # Download file content
            request = self.drive_service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            
            # Process according to file type
            text_content = ""
            
            if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
                # Process PDF file
                try:
                    pdf_reader = PyPDF2.PdfReader(file_content)
                    for page_num in range(len(pdf_reader.pages)):
                        text_content += pdf_reader.pages[page_num].extract_text() + "\n"
                except Exception as e:
                    return f"Error extracting PDF text: {str(e)}"
                        
            elif mime_type == "application/vnd.google-apps.document":
                # Google Doc - export as plain text
                text_content = self.drive_service.files().export(
                    fileId=file_id, mimeType='text/plain').execute().decode('utf-8')
                    
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.lower().endswith(".docx"):
                # Process DOCX file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                    temp_file.write(file_content.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    text_content = docx2txt.process(temp_file_path)
                except Exception as e:
                    text_content = f"Error processing DOCX file: {str(e)}"
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            elif mime_type == "text/plain" or file_name.lower().endswith((".txt", ".text")):
                # Plain text file
                text_content = file_content.getvalue().decode('utf-8', errors='replace')
                
            else:
                # Unsupported file type
                return f"Unsupported file type: {mime_type}. Please convert to PDF, DOCX, or TXT."
            
            return text_content
            
        except HttpError as error:
            print(f"Error accessing file {file_id}: {error}")
            return f"Error: {str(error)}"
        except Exception as e:
            print(f"Error processing file {file_id}: {e}")
            return f"Error: {str(e)}"
    
    def get_cv_content(self, cv_link: str) -> str:
        """
        Extract text content from a CV given its Google Drive link.
        
        Args:
            cv_link: Google Drive link to the CV document
            
        Returns:
            The extracted text content from the CV
        """
        # Extract file ID from link
        file_id = self.extract_file_id_from_drive_link(cv_link)
        
        if not file_id:
            return "Error: Could not extract file ID from the provided link."
        
        # Download and extract text
        return self.download_and_extract_text_from_drive_file(file_id)
    
    def get_talent_with_cv(self, row_number: int) -> Tuple[Dict[str, Any], str]:
        """
        Get a specific talent profile with their CV content.
        
        Args:
            row_number: The row number of the talent in the Talents Sheet
            
        Returns:
            A tuple containing the talent details and their CV content
        """
        talents_df = self.get_all_talents()
        
        # Find the specific talent
        talent_row = talents_df[talents_df['row_number'] == row_number]
        
        if talent_row.empty:
            return {}, "Error: Talent not found."
        
        # Convert row to dictionary
        talent_dict = talent_row.iloc[0].to_dict()
        
        # Get CV content if link exists
        cv_link = ""
        cv_column_name = talents_df.columns[self.cv_column_index] if self.cv_column_index < len(talents_df.columns) else None
        
        if cv_column_name and cv_column_name in talent_dict:
            cv_link = talent_dict[cv_column_name]
            
        cv_content = self.get_cv_content(cv_link) if cv_link else "No CV link provided."
        
        return talent_dict, cv_content
    
    def get_talent_model(self, row_number: int) -> Optional[CandidateProfile]:
        """
        Retrieve and validate talent details as a Pydantic model.
        
        Args:
            row_number: The row number of the talent in the Talents Sheet
            
        Returns:
            A validated CandidateProfile model or None if retrieval fails
        """
        talent_dict, cv_content = self.get_talent_with_cv(row_number)
        if not talent_dict:
            return None
            
        try:
            return parse_candidate_to_model(talent_dict, cv_content)
        except Exception as e:
            print(f"Error parsing candidate to model: {e}")
            return None
    
    def process_all_talents_with_cvs(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Process all talents and their CVs.
        
        Returns:
            A list of tuples, each containing talent details and CV content
        """
        talents_df = self.get_all_talents()
        results = []
        
        # Get CV column name
        cv_column_name = talents_df.columns[self.cv_column_index] if self.cv_column_index < len(talents_df.columns) else None
        
        if not cv_column_name:
            print(f"Warning: CV column index {self.cv_column_index} is out of range.")
            return []
        
        # Process each talent
        for index, row in talents_df.iterrows():
            talent_dict = row.to_dict()
            cv_link = row.get(cv_column_name, "")
            cv_content = self.get_cv_content(cv_link) if cv_link else "No CV link provided."
            results.append((talent_dict, cv_content))
            
        return results
    
    def get_all_talent_models(self) -> List[CandidateProfile]:
        """
        Process all talents and convert them to validated Pydantic models.
        
        Returns:
            A list of validated CandidateProfile models
        """
        all_talents_with_cvs = self.process_all_talents_with_cvs()
        talent_models = []
        
        for talent_dict, cv_content in all_talents_with_cvs:
            try:
                model = parse_candidate_to_model(talent_dict, cv_content)
                talent_models.append(model)
            except Exception as e:
                print(f"Error parsing candidate {talent_dict.get('Name', 'Unknown')} to model: {e}")
                # Continue processing other candidates even if one fails
                continue
                
        return talent_models

# Example usage
def main():
    # Path to your service account credentials JSON file
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    
    # Initialize the tool
    tool = TalentMatchingTool(credentials_path)
    
    # Example: Get job details from row 2 as a validated model
    job_row = 2
    job_model = tool.get_job_model(job_row)
    print(f"Job Model from row {job_row}:")
    if job_model:
        print(f"Title: {job_model.title}")
        print(f"Required skills: {', '.join([skill.name for skill in job_model.required_skills])}")
        print(f"Experience level: {job_model.experience_level}")
    else:
        print("Failed to retrieve job model")
    print("\n")
    
    # Example: Get talent with CV as a validated model
    talent_row = 2
    talent_model = tool.get_talent_model(talent_row)
    print(f"Talent Model from row {talent_row}:")
    if talent_model:
        print(f"Name: {talent_model.name}")
        print(f"Skills: {', '.join([skill.name for skill in talent_model.skills])}")
        print(f"CV content length: {len(talent_model.cv_content) if talent_model.cv_content else 0} characters")
    else:
        print("Failed to retrieve talent model")
    
if __name__ == "__main__":
    main()