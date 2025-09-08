"""
Query Classification Evaluator

Evaluates the accuracy of query type classification.
"""

import os
import sys

# Add path to import logger_config from happyLittleTreesOfKnowledge
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'happyLittleTreesOfKnowledge'))
from logger_config import setup_logger

logger = setup_logger(__name__)


def query_classification(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluate accuracy of query type classification.
    
    Tests whether the agent correctly identified the type of help the user needs
    (e.g., error_help, api_usage, code_explanation, etc.).
    
    Args:
        inputs: The input data passed to the agent
        outputs: The outputs from the agent being evaluated
        reference_outputs: The expected/reference outputs from the dataset
        
    Returns:
        bool: True if classification is correct, False otherwise.
    """
    try:
        expected_query_type = reference_outputs.get("expected_query_type")
        actual_query_type = outputs.get("query_type")
        
        # Direct classification accuracy
        classification_correct = expected_query_type == actual_query_type
        
        if not classification_correct:
            logger.debug(f"Classification mismatch - Expected: {expected_query_type}, Actual: {actual_query_type}")
        
        return classification_correct
        
    except Exception as e:
        logger.error(f"Error in query classification evaluator: {e}")
        return False