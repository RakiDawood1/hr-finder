"""
AutoGen-based Matching Engine for Talent Matching Tool

This module provides the core matching functionality using Microsoft's AutoGen framework
to match job requirements with candidate profiles.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import pandas as pd
from datetime import datetime
import os

# Import Pydantic models
from pydantic_models import JobRequirement, CandidateProfile, MatchResult

# Import AutoGen-based framework
from autogen_framework import AutoGenTalentMatcher

# Import talent matching tool for data access
from talent_matching_tool import TalentMatchingTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autogen_matching_engine")

class AutoGenMatchingEngine:
    """
    Engine for matching job requirements with candidate profiles using 
    the AutoGen-based multi-agent framework.
    """
    
    def __init__(self, tool: TalentMatchingTool, config_list=None, verbose=True):
        """
        Initialize the matching engine.
        
        Args:
            tool: Instance of TalentMatchingTool for data access
            config_list: Configuration for the LLM (if None, agents will use function calling only)
            verbose: Whether to display detailed agent conversations
        """
        self.tool = tool
        self.matcher = AutoGenTalentMatcher(config_list=config_list, verbose=verbose)
        logger.info("AutoGen Matching Engine initialized")
    
    def match_job_to_candidates(
        self,
        job_row: int,
        min_match_threshold: float = 0.3,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Match a job to suitable candidates using AutoGen agents.
        
        Args:
            job_row: Row number of the job in the Jobs Sheet
            min_match_threshold: Minimum match threshold (0.0 to 1.0)
            top_n: Maximum number of candidates to return
            
        Returns:
            Dictionary with match results
        """
        start_time = datetime.now()
        logger.info(f"Starting AutoGen matching process for job row {job_row}")
        
        # Step 1: Retrieve job details as a validated model
        job_model = self.tool.get_job_model(job_row)
        if not job_model:
            return {
                "success": False,
                "error": f"Failed to retrieve job model for row {job_row}",
                "job_row": job_row
            }
        
        logger.info(f"Retrieved job: {job_model.title}")
        
        # Step 2: Retrieve all candidate profiles as validated models
        all_candidates = self.tool.get_all_talent_models()
        if not all_candidates:
            return {
                "success": False,
                "error": "Failed to retrieve candidate profiles",
                "job_row": job_row,
                "job_title": job_model.title
            }
        
        logger.info(f"Retrieved {len(all_candidates)} candidate profiles")
        
        # Step 3: Use AutoGen framework to match candidates
        match_results, _ = self.matcher.match_candidates_to_job(
            job_model,
            all_candidates,
            min_match_threshold,
            top_n
        )
        
        # Step 4: Retrieve original row numbers for matched candidates
        matched_candidates_with_rows = self._get_row_numbers_for_matched_candidates(match_results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Step 5: Prepare result
        result = {
            "success": True,
            "job_row": job_row,
            "job_title": job_model.title,
            "total_candidates": len(all_candidates),
            "matched_candidates": len(match_results),
            "execution_time_seconds": execution_time,
            "matches": matched_candidates_with_rows,
            "match_details": match_results
        }
        
        logger.info(f"Completed AutoGen matching process. Found {len(match_results)} suitable candidates in {execution_time:.2f} seconds")
        return result
    
    def _get_row_numbers_for_matched_candidates(self, match_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve original row numbers for matched candidates.
        
        Args:
            match_results: List of dictionaries with match results
            
        Returns:
            List of dictionaries with candidate info and row numbers
        """
        # Get all candidates as a DataFrame for lookup
        candidates_df = self.tool.get_all_talents()
        
        results = []
        for match in match_results:
            candidate = match.get("candidate", {})
            candidate_name = candidate.get("name", "")
            candidate_email = candidate.get("email", "")
            
            # Look up the row number based on name and email
            row_number = None
            if candidates_df is not None and not candidates_df.empty:
                # Try to find by email if available (more reliable)
                if candidate_email:
                    matching_rows = candidates_df[candidates_df['Email'] == candidate_email]
                    if not matching_rows.empty:
                        row_number = matching_rows.iloc[0]['row_number']
                
                # If email doesn't match or not available, try by name
                if row_number is None:
                    matching_rows = candidates_df[candidates_df['Name'] == candidate_name]
                    if not matching_rows.empty:
                        row_number = matching_rows.iloc[0]['row_number']
            
            results.append({
                "name": candidate_name,
                "match_score": round(match.get("match_score", 0), 1),
                "row_number": row_number
            })
            
        return results


def main():
    """Example usage of the AutoGen matching engine."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials path from environment
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
    
    # Initialize the tool and engine
    tool = TalentMatchingTool(credentials_path)
    engine = AutoGenMatchingEngine(tool, config_list=None, verbose=True)
    
    # Example: Match job row 2 to candidates
    job_row = 2
    result = engine.match_job_to_candidates(job_row, top_n=5)
    
    if result["success"]:
        print(f"\nMatched candidates for {result['job_title']} (Job Row {job_row}):")
        print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
        print("-" * 50)
        
        for i, match in enumerate(result["match_details"], 1):
            print(f"{i}. {match['name']} - Match Score: {match['match_score']:.1f}")
            print(f"   Row Number: {next((r['row_number'] for r in result['matches'] if r['name'] == match['name']), 'Unknown')}")
            print(f"   Required Skills: {match['required_skills_matched']}")
            print(f"   Experience Match: {match['experience_match']}")
            print(f"   Location Match: {match['location_match']}")
            print(f"   {match['explanation']}")
            print()
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()