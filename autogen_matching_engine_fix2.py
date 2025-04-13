"""
AutoGen-based Matching Engine for Talent Matching Tool

This module provides the core matching functionality using a simplified framework
to match job requirements with candidate profiles while making the process visible.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import pandas as pd
from datetime import datetime
import os

# Import Pydantic models
from pydantic_models import JobRequirement, CandidateProfile, MatchResult

# Import AutoGen-based framework
from autogen_framework_fix2 import AutoGenTalentMatcher

# Try to import Gemini integration
try:
    from gemini_integrations import get_gemini_config_from_env
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import talent matching tool for data access
from talent_matching_tool_fix2 import TalentMatchingTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autogen_matching_engine")

class AutoGenMatchingEngine:
    """
    Engine for matching job requirements with candidate profiles using 
    a multi-step evaluation process with detailed output.
    """
    
    def __init__(self, tool: TalentMatchingTool, config_list=None, verbose=True, use_gemini=False, debug=False):
        """
        Initialize the matching engine.
        
        Args:
            tool: Instance of TalentMatchingTool for data access
            config_list: Configuration for the LLM (if None, agents will use function calling only)
            verbose: Whether to display detailed matching process
            use_gemini: Whether to use Gemini API (requires GEMINI_API_KEY in environment)
            debug: Enable debug logging for detailed troubleshooting
        """
        self.tool = tool
        self.verbose = verbose
        self.debug = debug
        self.matcher = AutoGenTalentMatcher(
            tool=tool,
            config_list=config_list, 
            verbose=verbose, 
            use_gemini=use_gemini,
            debug=debug
        )
        
        # Set up logging based on debug flag
        log_level = logging.DEBUG if debug else logging.INFO
        logging.getLogger("autogen_matching_engine").setLevel(log_level)
        
        # Log initialization
        logger.info("AutoGen Matching Engine initialized with enhanced matching process")
        if debug:
            logger.debug("Debug mode enabled - detailed logging will be shown")
    
    def match_job_to_candidates(
        self,
        job_row: int,
        min_match_threshold: float = 0.3,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Match a job to suitable candidates using a two-stage filtering process:
        1. First filtering by job preference (Column F)
        2. Then detailed analysis only on pre-filtered candidates
        
        Args:
            job_row: Row number of the job in the Jobs Sheet
            min_match_threshold: Minimum match threshold (0.0 to 1.0)
            top_n: Maximum number of candidates to return
            
        Returns:
            Dictionary with match results
        """
        start_time = datetime.now()
        logger.info(f"Starting matching process for job row {job_row}")
        
        # Step 1: Retrieve job details as a validated model
        job_model = self.tool.get_job_model(job_row)
        if not job_model:
            return {
                "success": False,
                "error": f"Failed to retrieve job model for row {job_row}",
                "job_row": job_row
            }
        
        if self.debug:
            logger.debug(f"Job Model details:")
            logger.debug(f"Title: {job_model.title}")
            logger.debug(f"Required Skills: {[s.name for s in job_model.required_skills]}")
            logger.debug(f"Experience Level: {job_model.experience_level}")
            logger.debug(f"Min Years Experience: {job_model.min_years_experience}")
        
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
        
        if self.debug:
            logger.debug(f"\nCandidate details:")
            for candidate in all_candidates:
                logger.debug(f"\nCandidate: {candidate.name}")
                logger.debug(f"Skills: {[s.name for s in candidate.skills]}")
                logger.debug(f"Years Experience: {candidate.years_of_experience}")
                logger.debug(f"CV Content Length: {len(candidate.cv_content) if candidate.cv_content else 0}")
        
        logger.info(f"Retrieved {len(all_candidates)} candidate profiles")
        
        # Step 3: First stage - Filter by job preference (Column F)
        try:
            # Print banner for improved visibility
            print("\n" + "="*80)
            print(f"TALENT MATCHING PROCESS: {job_model.title}")
            print("="*80)
            
            # Get job preference from Column F
            job_preference = job_model.job_preference if hasattr(job_model, 'job_preference') else None
            
            # Filter candidates by job preference
            pre_filtered_candidates = []
            if job_preference:
                logger.info(f"Filtering candidates by job preference: {job_preference}")
                for candidate in all_candidates:
                    if hasattr(candidate, 'preferred_job_types') and job_preference in candidate.preferred_job_types:
                        pre_filtered_candidates.append(candidate)
                logger.info(f"Pre-filtered {len(pre_filtered_candidates)} candidates by job preference")
            else:
                logger.info("No job preference specified, using all candidates")
                pre_filtered_candidates = all_candidates
            
            # Step 4: Second stage - Detailed analysis on pre-filtered candidates
            match_results, _ = self.matcher.match_candidates_to_job(
                job_model,
                pre_filtered_candidates,
                min_match_threshold,
                top_n
            )
            
            # Log successful matching
            logger.info(f"Successfully matched {len(match_results)} candidates to job {job_model.title}")
            
        except Exception as e:
            logger.error(f"Error during matching process: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Matching error: {str(e)}",
                "job_row": job_row,
                "job_title": job_model.title
            }
        
        # Step 5: Retrieve original row numbers for matched candidates
        matched_candidates_with_rows = self._get_row_numbers_for_matched_candidates(match_results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Step 6: Prepare result
        result = {
            "success": True,
            "job_row": job_row,
            "job_title": job_model.title,
            "total_candidates": len(all_candidates),
            "pre_filtered_candidates": len(pre_filtered_candidates),
            "matched_candidates": len(match_results),
            "execution_time_seconds": execution_time,
            "matches": matched_candidates_with_rows,
            "match_details": match_results
        }
        
        if self.debug:
            logger.debug("\nFinal Match Results:")
            for match in match_results:
                logger.debug(f"\nCandidate: {match.get('name', '')}")
                logger.debug(f"Match Score: {match.get('match_score', 0):.1f}")
                logger.debug(f"Required Skills Matched: {match.get('required_skills_matched', '')}")
                logger.debug(f"Experience Match: {match.get('experience_match', '')}")
        
        logger.info(f"Completed matching process. Found {len(match_results)} suitable candidates in {execution_time:.2f} seconds")
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
                    # Case-insensitive name matching
                    matching_rows = candidates_df[candidates_df['Name'].str.lower() == candidate_name.lower()]
                    if not matching_rows.empty:
                        row_number = matching_rows.iloc[0]['row_number']
                        
                    # Try partial name matching if exact match fails
                    if row_number is None and len(candidate_name.split()) > 1:
                        # Try matching first and last name separately
                        name_parts = candidate_name.lower().split()
                        for part in name_parts:
                            if len(part) > 3:  # Only try with longer name parts
                                matching_rows = candidates_df[candidates_df['Name'].str.lower().str.contains(part)]
                                if not matching_rows.empty:
                                    row_number = matching_rows.iloc[0]['row_number']
                                    break
            
            results.append({
                "name": candidate_name,
                "match_score": round(match.get("match_score", 0), 1),
                "row_number": row_number
            })
            
        return results

    def _filter_candidates(self, job: Dict[str, Any], candidates: List[CandidateProfile], 
                           min_match_threshold: float = 0.5) -> List[MatchResult]:
        """Filter candidates based on detailed criteria including experience and CV analysis."""
        logger.info(f"Performing detailed analysis on {len(candidates)} candidates")
        matched_candidates = []
        
        # Define role expertise based on job details
        role_expertise = self._research_role_requirements(
            job.get("title", ""),
            job.get("important_qualities", "")
        )

        for candidate in candidates:
            logger.debug(f"\nEvaluating candidate: {candidate.name} (Row: {getattr(candidate, 'row_number', 'N/A')})")
            
            # Basic experience check (if required)
            min_exp = job.get("min_years_experience", 0) or 0
            candidate_exp = candidate.years_of_experience or 0
            if min_exp > 0 and candidate_exp < min_exp:
                logger.info(f"  Skipping {candidate.name} due to insufficient experience ({candidate_exp} < {min_exp})")
                continue
            
            # Evaluate CV content
            cv_analysis = self._evaluate_candidate_cv(candidate, role_expertise)
            overall_score = self._calculate_overall_score(cv_analysis, candidate)

            if overall_score >= min_match_threshold:
                matched_candidates.append(MatchResult(
                    candidate_id=candidate.candidate_id,
                    candidate_name=candidate.name,
                    score=overall_score,
                    details=cv_analysis,
                    row_number=getattr(candidate, 'row_number', None) # Include row number
                ))
            else:
                logger.info(f"  Candidate {candidate.name} did not meet threshold ({overall_score:.2f} < {min_match_threshold}) ")
        
        # Sort by score descending
        matched_candidates.sort(key=lambda x: x.score, reverse=True)
        
        return matched_candidates

    def _calculate_overall_score(self, cv_analysis: Dict[str, Any], candidate: CandidateProfile) -> float:
        """Calculate overall score based on CV analysis and candidate details."""
        cv_score = cv_analysis.get("overall_cv_score", 0.1)
        skill_score = self._evaluate_skills_match(candidate, cv_analysis.get("inferred_skills", []))
        
        return (cv_score * 0.6) + (skill_score * 0.4)

    def match_candidates_to_job(self, job: JobRequirement, all_candidates: List[CandidateProfile],
                              min_match_threshold: float = 0.3, top_n: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Match candidates to a job using the enhanced two-stage filtering framework."""
        logger.info(f"Starting candidate matching process for job: {job.title}")
        
        job_dict = job.model_dump()
        candidate_dicts = [c.model_dump() for c in all_candidates]
        
        # STAGE 1: Job Preference Filtering (Coordinator Agent)
        logger.info(f"Performing job preference filtering on {len(all_candidates)} candidates")
        print("\n" + "="*80)
        print(f"TALENT MATCHING PROCESS: {job.title}")
        print("="*80)

        # First filter by job preference (using original objects)
        job_preference_filtered = self._filter_candidates_by_job_preference(job_dict, all_candidates)

        # STAGE 2: Detailed evaluation with CV analysis (only on job preference filtered candidates)
        try:
            logger.info(f"Performing detailed analysis on {len(job_preference_filtered)} job preference filtered candidates")
            print("\nSTAGE 2: HR MANAGER AGENT - DETAILED CANDIDATE EVALUATION")
            print("-"*80)

            # Apply second-stage filtering and full analysis only to candidates passing first filter
            filtered_candidates = self._filter_candidates(job_dict, job_preference_filtered, min_match_threshold)

            # Get final rankings with detailed explanations
            match_results = self._rank_candidates_with_detail(job_dict, filtered_candidates, top_n)

            logger.info(f"Successfully matched and ranked {len(match_results)} candidates")
            print(f"\nSuccessfully matched and ranked {len(match_results)} candidates")

        except Exception as e:
            logger.error(f"Error during detailed matching process: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return [], []
        
        return match_results, []

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