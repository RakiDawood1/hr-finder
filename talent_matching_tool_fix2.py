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
        
        # Set CV column index (column N = 13)
        self.cv_column_index = int(os.getenv('CV_COLUMN_INDEX', '13'))
        
        # Validate and adjust CV column index
        self._validate_cv_column()
        
        # Cache for column indices 
        self._talents_col_indices = None
        
    def _validate_cv_column(self):
        """Validate and adjust CV column index if necessary."""
        try:
            # Get sheet headers
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.talents_sheet_id,
                range=f"{self.talents_range.split('!')[0]}!1:1"  # Get first row only
            ).execute()
            
            headers = result.get('values', [[]])[0]
            print("\nValidating CV column configuration:")
            print(f"Total columns found: {len(headers)}")
            print(f"Current CV column index: {self.cv_column_index}")
            
            # Look for CV column
            cv_column = None
            for i, header in enumerate(headers):
                if header.lower() == 'cv':
                    cv_column = i
                    break
            
            if cv_column is not None:
                print(f"Found CV column at index {cv_column}")
                if cv_column != self.cv_column_index:
                    print(f"Adjusting CV column index from {self.cv_column_index} to {cv_column}")
                    self.cv_column_index = cv_column
            else:
                # If no CV column found, try to find similar column names
                cv_like_columns = [(i, h) for i, h in enumerate(headers) 
                                 if 'cv' in h.lower() or 'resume' in h.lower()]
                if cv_like_columns:
                    print(f"CV column not found, but found similar columns:")
                    for i, h in cv_like_columns:
                        print(f"  Column {i}: {h}")
                else:
                    print("Warning: No CV column found in headers")
            
            # Print column mapping for verification
            print("\nColumn mapping:")
            for i, header in enumerate(headers):
                print(f"Column {i} ({chr(65+i)}): {header}")
                
        except Exception as e:
            print(f"Error validating CV column: {e}")
            print("Using default CV column index")
        
    def _get_talents_column_indices(self):
        """Get important column indices for the talents sheet to avoid hard-coding."""
        if self._talents_col_indices is not None:
            return self._talents_col_indices
            
        result = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=self.talents_sheet_id,
            range=self.talents_range
        ).execute()
        
        values = result.get('values', [])
        headers = values[0] if values else []
        
        # Find key column indices
        indices = {
            'name': next((i for i, h in enumerate(headers) if h and "Name" in h), 0),
            'experience': next((i for i, h in enumerate(headers) if h and "Experience" in h), 6),
            'position': next((i for i, h in enumerate(headers) if h and "Position" in h), 5),
            'job_preference': next((i for i, h in enumerate(headers) if h and ("Applying" in h or "Job" in h and "Looking" in h)), 5),
            'location': next((i for i, h in enumerate(headers) if h and "Location" in h), 4),
            'cv': self.cv_column_index
        }
        
        print(f"Talent Sheet Column Indices: {indices}")
        self._talents_col_indices = indices
        return indices
        
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
            # Get raw data from the sheet to access columns by index
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.jobs_sheet_id,
                range=self.jobs_range
            ).execute()
            
            values = result.get('values', [])
            if values and len(values) > row_number - 1:
                headers = values[0]
                row_data = values[row_number - 1]
                
                # Map Column I (index 8) - "What's most Important in a candidate"
                if len(row_data) > 8:
                    important_qualities = row_data[8]
                    job_dict["important_qualities"] = important_qualities
                    print(f"Important qualities from Column I: {important_qualities}")
                
                # Use column E as the job title (index 4)
                if len(headers) > 4 and len(row_data) > 4:
                    col_e_header = headers[4]  # "Roles Looking to Fill"
                    col_e_value = row_data[4]
                    
                    if col_e_value:
                        title_key = next((key for key in job_dict.keys() if key.lower() == "title"), None)
                        if title_key:
                            job_dict[title_key] = col_e_value
                        else:
                            job_dict["Title"] = col_e_value
                
                # Map "Skill and Requirement" to RequiredSkills
                col_g_index = 6  # Column G (index 6)
                if len(headers) > col_g_index and len(row_data) > col_g_index:
                    col_g_header = headers[col_g_index]  # Should be "Skill and Requirement"
                    col_g_value = row_data[col_g_index]
                    
                    if col_g_value and "Skill" in col_g_header:
                        job_dict["RequiredSkills"] = col_g_value
                
                # Add appropriate experience level based on column F
                col_f_index = 5  # Column F (index 5)
                if len(headers) > col_f_index and len(row_data) > col_f_index:
                    col_f_header = headers[col_f_index]  # Should be "Experience Level"
                    col_f_value = row_data[col_f_index]
                    
                    if col_f_value and "Experience" in col_f_header:
                        job_dict["ExperienceLevel"] = col_f_value
                        
                        # Try to extract years of experience
                        years_match = re.search(r'(\d+)[\s-]*(\d*)', col_f_value)
                        if years_match:
                            min_years = int(years_match.group(1))
                            job_dict["YearsExperience"] = str(min_years)
        
        except Exception as e:
            print(f"Warning: Error while enhancing job data: {e}")
        
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
            
        # Clean the link
        drive_link = drive_link.strip().replace('\n', '').replace(' ', '')
        print(f"\nExtracting file ID from link: {drive_link}")
        
        # Define patterns - order matters!
        patterns = [
            # Pattern for Google Docs with parameters
            r"https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)/edit",
            
            # Pattern for /file/d/ format
            r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
            
            # Pattern for direct document links
            r"https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)",
            
            # Pattern for spreadsheets
            r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)",
            
            # Pattern for presentations
            r"https://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
            
            # Pattern for folders
            r"https://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
            
            # Pattern for direct download links
            r"https://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_link)
            if match:
                file_id = match.group(1)
                # File IDs are typically 33 characters - validate length
                if 25 <= len(file_id) <= 50:  # Being a bit flexible with length
                    print(f"Extracted file ID: {file_id}")
                    return file_id.split('?')[0]  # Remove any trailing parameters
        
        print(f"Warning: Could not extract file ID using standard patterns")
        return ""

    def _validate_file_id(self, file_id: str) -> bool:
        """Validate if a file ID is likely to be correct."""
        if not file_id:
            return False
            
        # Basic validation for Google Drive file IDs
        # - Typically 33 characters
        # - Contains letters, numbers, underscores, and hyphens
        # - No special characters
        if 25 <= len(file_id) <= 50 and re.match(r'^[a-zA-Z0-9_-]+$', file_id):
            return True
        return False

    def extract_cv_content(self, drive_link: str) -> str:
        """
        Download and extract text content from a CV file.
        
        Args:
            drive_link: Google Drive link to the CV file
            
        Returns:
            Extracted text content from the CV
        """
        if not drive_link:
            print("Warning: Empty drive link provided")
            return ""
            
        try:
            # Extract file ID from the drive link
            file_id = self.extract_file_id_from_drive_link(drive_link)
            if not file_id or not self._validate_file_id(file_id):
                print(f"Warning: Invalid or no file ID extracted from link: {drive_link}")
                return ""
                
            print(f"Processing file ID: {file_id}")
                
            # Get file metadata
            try:
                print("Getting file metadata...")
                file_metadata = self.drive_service.files().get(
                    fileId=file_id,
                    fields='mimeType,name'
                ).execute()
                
                mime_type = file_metadata.get('mimeType', '')
                file_name = file_metadata.get('name', '')
                print(f"File: {file_name} ({mime_type})")
                
            except Exception as e:
                print(f"Error getting file metadata: {str(e)}")
                return ""
            
            text = ""
            # Handle different file types
            if 'application/vnd.google-apps.document' in mime_type:
                # Native Google Doc
                print("Exporting Google Doc as text...")
                request = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
                content = request.execute()
                text = content.decode('utf-8')
                
            elif 'application/pdf' in mime_type:
                # For PDF files
                print("Downloading PDF content...")
                request = self.drive_service.files().get_media(fileId=file_id)
                content = io.BytesIO(request.execute())
                
                try:
                    # Try to use pdfplumber first if available
                    try:
                        import pdfplumber
                        with pdfplumber.open(content) as pdf:
                            text_parts = []
                            for page in pdf.pages:
                                text_parts.append(page.extract_text() or '')
                            text = '\n\n'.join(text_parts)
                    except ImportError:
                        # Fall back to PyPDF2
                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                            temp_pdf.write(content.getvalue())
                            temp_pdf_path = temp_pdf.name
                        
                        try:
                            pdf_reader = PyPDF2.PdfReader(temp_pdf_path)
                            text_parts = []
                            for page in pdf_reader.pages:
                                text_parts.append(page.extract_text() or '')
                            text = '\n\n'.join(text_parts)
                        finally:
                            if os.path.exists(temp_pdf_path):
                                os.unlink(temp_pdf_path)
                except Exception as pdf_error:
                    print(f"Error extracting PDF content: {str(pdf_error)}")
                    return ""
                    
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in mime_type:
                # For DOCX files - download and process using docx2txt
                print("Downloading and processing DOCX file...")
                request = self.drive_service.files().get_media(fileId=file_id)
                content = io.BytesIO(request.execute())
                
                # Save as temporary file
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                    temp_file.write(content.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    # Extract text using docx2txt
                    text = docx2txt.process(temp_file_path)
                    print(f"Successfully extracted text from DOCX using docx2txt")
                except Exception as docx_error:
                    print(f"Error with docx2txt, trying alternative method: {str(docx_error)}")
                    try:
                        # Alternative approach using python-docx
                        import docx
                        doc = docx.Document(temp_file_path)
                        text = "\n".join([para.text for para in doc.paragraphs])
                        print(f"Successfully extracted text from DOCX using python-docx")
                    except Exception as alt_error:
                        print(f"Error extracting DOCX content: {str(alt_error)}")
                        return ""
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            else:
                # For other file types, try direct download
                print(f"Attempting direct download of {mime_type} file...")
                try:
                    request = self.drive_service.files().get_media(fileId=file_id)
                    content = request.execute()
                    text = content.decode('utf-8', errors='ignore')
                except Exception as e:
                    print(f"Error downloading file content: {str(e)}")
                    return ""
            
            # Clean and validate the extracted text
            if text:
                text = text.strip()
                print(f"Successfully extracted {len(text)} characters")
                # Print first 100 chars as debug info
                if len(text) > 100:
                    print(f"Text preview: {text[:100]}...")
                return text
            else:
                print("Warning: No text content extracted")
                return ""
                
        except Exception as e:
            print(f"Error extracting CV content: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return ""
    
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
        return self.extract_cv_content(cv_link)
    
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
            print(f"Warning: No talent found for row {row_number}")
            return {}, "Error: Talent not found."
        
        # Convert row to dictionary
        talent_dict = talent_row.iloc[0].to_dict()
        
        # Debug info
        print("\nDEBUG: Talent CV Data")
        print("-" * 50)
        print(f"Row number: {row_number}")
        
        # Get CV content if link exists
        cv_link = ""
        cv_column_name = None
        
        # Print column names and indices for debugging
        print("\nColumn Index Check:")
        for i, col in enumerate(talents_df.columns):
            print(f"Column {i}: {col}")
        
        cv_column_idx = self.cv_column_index
        print(f"\nCV Column Index: {cv_column_idx}")
        
        if cv_column_idx < len(talents_df.columns):
            cv_column_name = talents_df.columns[cv_column_idx]
            print(f"CV Column Name: {cv_column_name}")
        else:
            print("Warning: CV column index out of range")
        
        if cv_column_name and cv_column_name in talent_dict:
            cv_link = talent_dict[cv_column_name]
            print(f"\nCV Link for {talent_dict.get('name', 'Unknown')}: {cv_link}")
        else:
            print(f"\nWarning: No CV column found or CV link missing")
            
        cv_content = ""
        if cv_link:
            print("\nAttempting to extract CV content...")
            cv_content = self.extract_cv_content(cv_link)
            print(f"Extracted CV content length: {len(cv_content)}")
            print("First 100 characters of CV content:")
            print("-" * 50)
            print(cv_content[:100])
            print("-" * 50)
        else:
            cv_content = "No CV link provided."
            print("No CV link to process")
        
        return talent_dict, cv_content

    def process_all_talents_with_cvs(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Process all talents and their CVs.
        
        Returns:
            A list of tuples, each containing talent details and CV content
        """
        talents_df = self.get_all_talents()
        results = []
        
        print("\nDEBUG: Processing all talents")
        print("-" * 50)
        
        # Get CV column name
        cv_column_name = talents_df.columns[self.cv_column_index] if self.cv_column_index < len(talents_df.columns) else None
        
        if not cv_column_name:
            print(f"Warning: CV column index {self.cv_column_index} is out of range")
            print(f"Available columns: {list(talents_df.columns)}")
            return []
        
        print(f"Using CV column: {cv_column_name}")
        
        # Process each talent
        for index, row in talents_df.iterrows():
            talent_dict = row.to_dict()
            talent_name = talent_dict.get('Name', 'Unknown')
            cv_link = row.get(cv_column_name, "")
            
            print(f"\nProcessing {talent_name}:")
            print(f"CV Link: {cv_link}")
            
            if cv_link:
                print("Extracting CV content...")
                cv_content = self.extract_cv_content(cv_link)
                print(f"Extracted content length: {len(cv_content)}")
            else:
                cv_content = "No CV link provided."
                print("No CV link to process")
                
            results.append((talent_dict, cv_content))
            
        return results
    
    def _enhance_talent_from_sheet(self, talent_dict: Dict[str, Any], row_data: List, headers: List) -> Dict[str, Any]:
        """
        Helper function to enhance talent data using raw sheet data.
        
        Args:
            talent_dict: Dictionary with current talent data
            row_data: Raw row data from the sheet
            headers: Sheet headers
            
        Returns:
            Enhanced talent dictionary
        """
        col_indices = self._get_talents_column_indices()
        
        # Set experience data from the correct column
        exp_col_idx = col_indices['experience']
        if len(row_data) > exp_col_idx:
            exp_value = row_data[exp_col_idx]
            if exp_value:
                # Extract years as a number
                years_match = re.search(r'(\d+)', exp_value)
                if years_match:
                    # Direct assignment
                    years_exp = int(years_match.group(1))
                    talent_dict["YearsExperience"] = years_exp
                    talent_dict["years_of_experience"] = years_exp  # Add both forms
                    print(f"Set experience for {talent_dict.get('Name', 'unknown')}: {years_exp} years")
        
        # Set position data from the correct column (job title)
        pos_col_idx = col_indices['position']
        if len(row_data) > pos_col_idx:
            pos_value = row_data[pos_col_idx]
            if pos_value:
                # Look for a key that could contain job title
                title_keys = ["CurrentTitle", "JobTitle", "Current Title", "Job Title", "Title", "Position"]
                found_key = next((key for key in title_keys if key in talent_dict), None)
                
                if found_key:
                    talent_dict[found_key] = pos_value
                else:
                    talent_dict["CurrentTitle"] = pos_value
        
        # Set job preference data from the correct column (what job they're applying for)
        job_pref_col_idx = col_indices['job_preference']
        if len(row_data) > job_pref_col_idx:
            job_pref_value = row_data[job_pref_col_idx]
            if job_pref_value:
                talent_dict["position_preference"] = job_pref_value
                talent_dict["jobs_applying_for"] = job_pref_value
                print(f"Set job preference for {talent_dict.get('Name', 'unknown')}: '{job_pref_value}'")
        
        # Set skills data from the correct column
        skills_col_idx = col_indices['skills']
        if len(row_data) > skills_col_idx:
            skills_value = row_data[skills_col_idx]
            if skills_value:
                # Look for a key that could contain skills
                skills_key = next((key for key in talent_dict.keys() if key.lower() == "skills"), None)
                if skills_key:
                    talent_dict[skills_key] = skills_value
                else:
                    talent_dict["Skills"] = skills_value
        
        return talent_dict
    
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
            # Get raw data from the sheet to enhance talent data
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.talents_sheet_id,
                range=self.talents_range
            ).execute()
            
            values = result.get('values', [])
            if values:
                headers = values[0]
                row_idx = next((i for i, row in enumerate(values) if len(row) > 0 and 
                               i > 0 and int(row_number) == i+1), None)
                
                if row_idx is not None:
                    row_data = values[row_idx]
                    talent_dict = self._enhance_talent_from_sheet(talent_dict, row_data, headers)
        
        except Exception as e:
            print(f"Warning: Error while enhancing talent data: {e}")
            
        try:
            model = parse_candidate_to_model(talent_dict, cv_content=cv_content, row_number=row_number)
            if model.years_of_experience is None or model.years_of_experience == 0:
                # Double-check the years_of_experience
                years_exp = talent_dict.get("YearsExperience") or talent_dict.get("years_of_experience")
                if years_exp:
                    model.years_of_experience = float(years_exp)
                    print(f"Fixed years of experience for {model.name}: {model.years_of_experience}")
            return model
        except Exception as e:
            print(f"Error parsing candidate to model: {e}")
            return None
    
    def get_all_talent_models(self) -> List[CandidateProfile]:
        """Retrieve all talent data and parse into CandidateProfile models."""
        try:
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.talents_sheet_id,
                range=self.talents_range
            ).execute()
            
            values = result.get('values', [])
            
            if not values or len(values) < 2:
                print("Warning: No talent data found or only headers.")
                return []
            
            headers = values[0]
            talent_data = values[1:]
            
            col_indices = self._get_talents_column_indices()
            
            print("\nDEBUG: Processing talent models (without upfront CV fetch)")
            print("--------------------------------------------------")
            print(f"Found {len(talent_data)} candidates in sheet")
            print(f"CV column index: {col_indices['cv']}")
            
            talent_models = []
            for i, row in enumerate(talent_data):
                row_number = i + 2  # Sheet rows are 1-indexed, data starts at row 2
                try:
                    print(f"\nProcessing candidate {row_number}: {row[col_indices['name']] if len(row) > col_indices['name'] else 'Unknown'}")
                    
                    # Prepare data dictionary from the row
                    talent_dict = {}
                    for idx, header in enumerate(headers):
                        if idx < len(row):
                            talent_dict[header] = row[idx]
                        else:
                            talent_dict[header] = "" # Handle rows with fewer columns than headers
                    
                    # Get CV link if available
                    cv_link = row[col_indices['cv']] if len(row) > col_indices['cv'] else None
                    if cv_link:
                        print(f"CV Link stored: {cv_link}")
                    
                    # Prepare data for Pydantic model
                    name_value = row[col_indices.get('name', 0)] if len(row) > col_indices.get('name', 0) else ""
                    print(f"Name from sheet: '{name_value}'")  # Debug output to verify name
                    
                    candidate_data = {
                        "name": name_value,
                        # Initialize skills as empty; they will be populated by CV analysis later
                        "skills": [], 
                        "years_of_experience": row[col_indices.get('experience', 6)] if len(row) > col_indices.get('experience', 6) else 0,
                        "current_title": row[col_indices.get('position', 5)] if len(row) > col_indices.get('position', 5) else "",
                        "position_preference": row[col_indices.get('job_preference', 5)] if len(row) > col_indices.get('job_preference', 5) else "",
                        "jobs_applying_for": row[col_indices.get('job_preference', 5)] if len(row) > col_indices.get('job_preference', 5) else "", # Use same column
                        "current_location": row[col_indices.get('location', 4)] if len(row) > col_indices.get('location', 4) else "",
                        "cv_link": cv_link,
                        "row_number": row_number
                        # Add other fields as needed, ensuring they exist in col_indices and row
                    }

                    # Debug: Print specific field assignments
                    if len(row) > col_indices.get('experience', 6):
                        print(f"Set experience for {candidate_data['name']}: {candidate_data['years_of_experience']}")
                    if len(row) > col_indices.get('job_preference', 5):
                        print(f"Set job preference for {candidate_data['name']}: '{candidate_data['position_preference']}'")

                    print("\nDEBUG: Parsing candidate to model")
                    print("--------------------------------------------------")
                    
                    # Parse and validate
                    # Pass only candidate_data dict and cv_content=None
                    candidate_model = parse_candidate_to_model(candidate_data, cv_content=None)
                    
                    if candidate_model:
                        print(f"Processing candidate: {candidate_model.name}")
                        print(f"CV content length: {len(candidate_model.cv_content) if candidate_model.cv_content else 0}")
                        print(f"Years of experience: {candidate_model.years_of_experience}")
                        talent_models.append(candidate_model)
                except Exception as e:
                    print(f"Error processing candidate at row {row_number}: {str(e)}")
                    import traceback
                    print(f"Stack trace: {traceback.format_exc()}")
                    continue # Skip this candidate and move to the next
                    
            print(f"\nProcessed {len(talent_models)} candidate models (CV content fetch deferred)")
            return talent_models
        except HttpError as error:
            print(f"An HTTP error occurred: {error}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return []

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
        print(f"Years Experience: {talent_model.years_of_experience}")
        print(f"Job Preference: {getattr(talent_model, 'jobs_applying_for', 'Not specified')}")
        print(f"CV content length: {len(talent_model.cv_content) if talent_model.cv_content else 0} characters")
    else:
        print("Failed to retrieve talent model")

if __name__ == "__main__":
    main()