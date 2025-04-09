"""
AutoGen-based Talent Matching Tool - Main Script

This script provides a command-line interface to the Talent Matching Tool using
Microsoft's AutoGen for the multi-agent framework.
"""

import os
import argparse
import json
from typing import Dict, Any
from dotenv import load_dotenv
from pprint import pprint
import logging

# Import the main components
from talent_matching_tool import TalentMatchingTool
from autogen_matching_engine import AutoGenMatchingEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autogen_talent_matching")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='AutoGen-based Talent Matching Tool')
    parser.add_argument('--job', type=int, help='Job row number to find candidates for')
    parser.add_argument('--threshold', type=float, default=0.3, 
                      help='Minimum match threshold (0.0 to 1.0)')
    parser.add_argument('--top', type=int, default=10, 
                      help='Maximum number of candidates to return')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save JSON output (optional)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose agent conversations')
    parser.add_argument('--llm-key', type=str, default=None,
                      help='API key for LLM (optional, leave empty to use function-only agents)')
    return parser.parse_args()

def setup_environment():
    """Set up the environment and configurations."""
    # Load environment variables from .env file
    load_dotenv(verbose=True)
    
    # Check for required environment variables
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'JOBS_SHEET_ID',
        'TALENTS_SHEET_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with these variables or set them in your environment.")
        return False
    
    return True

def get_llm_config(api_key=None):
    """Get LLM configuration for AutoGen (optional, can be None for function-only agents)."""
    if not api_key:
        return None
        
    return {
        "config_list": [{"model": "gpt-4", "api_key": api_key}],
        "cache_seed": 42
    }

def find_candidates_for_job(job_row: int, min_threshold: float = 0.3, top_n: int = 10, 
                          llm_key: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Find suitable candidates for the specified job using AutoGen.
    
    Args:
        job_row: Row number of the job in the Jobs Sheet
        min_threshold: Minimum match threshold (0.0 to 1.0)
        top_n: Maximum number of candidates to return
        llm_key: API key for LLM (optional)
        verbose: Whether to show detailed agent conversations
        
    Returns:
        Dictionary with match results
    """
    # Get credentials path from environment
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Initialize the main components
    tool = TalentMatchingTool(credentials_path)
    
    # Get LLM config if API key is provided
    config_list = get_llm_config(llm_key)
    
    # Initialize the AutoGen-based engine
    engine = AutoGenMatchingEngine(tool, config_list=config_list, verbose=verbose)
    
    # Match job to candidates
    return engine.match_job_to_candidates(job_row, min_threshold, top_n)

def display_results(results: Dict[str, Any]):
    """
    Display match results in a user-friendly format.
    
    Args:
        results: Dictionary with match results
    """
    if not results["success"]:
        print(f"Error: {results['error']}")
        return
    
    print("\n" + "=" * 50)
    print(f"JOB MATCHING RESULTS: {results['job_title']} (Row {results['job_row']})")
    print("=" * 50)
    print(f"Total candidates analyzed: {results['total_candidates']}")
    print(f"Candidates matched: {results['matched_candidates']}")
    print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
    print("-" * 50)
    
    if not results["match_details"]:
        print("No suitable candidates found.")
        return
    
    print("TOP MATCHING CANDIDATES:")
    for i, match in enumerate(results["match_details"], 1):
        # Find row number from matches list
        row_number = next((r.get('row_number', 'Unknown') for r in results['matches'] if r.get('name', '') == match.get('name', '')), 'Unknown')
        
        print(f"\n{i}. {match.get('name', '')} (Row {row_number})")
        print(f"   Match Score: {match.get('match_score', 0):.1f}/100")
        print(f"   Required Skills: {match.get('required_skills_matched', '0%')}")
        print(f"   Experience Match: {match.get('experience_match', 'No')}")
        print(f"   Location Match: {match.get('location_match', 'No')}")
        print(f"   Summary: {match.get('explanation', '')}")
    
    print("\n" + "=" * 50)

def save_results_to_file(results: Dict[str, Any], output_path: str):
    """
    Save match results to a JSON file.
    
    Args:
        results: Dictionary with match results
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    """Main entry point for the AutoGen-based Talent Matching Tool."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment setup
    if not setup_environment():
        return 1
    
    # Validate arguments
    if not args.job:
        logger.error("Please specify a job row number with --job")
        return 1
    
    # Find matching candidates
    logger.info(f"Finding candidates for job row {args.job} using AutoGen")
    results = find_candidates_for_job(
        args.job, 
        args.threshold, 
        args.top,
        args.llm_key,
        args.verbose
    )
    
    # Display results
    display_results(results)
    
    # Save results to file if requested
    if args.output:
        save_results_to_file(results, args.output)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)