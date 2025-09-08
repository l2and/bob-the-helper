"""
Confidence Calibration Evaluator

Evaluates how well-calibrated the agent's confidence scores are for understanding user intent.
"""

import logging

logger = logging.getLogger(__name__)


def confidence_calibration(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
    """
    Evaluate how well-calibrated the agent's confidence scores are for understanding user intent.
    
    Tests whether the agent's overall_confidence appropriately reflects the clarity/ambiguity 
    of the user's request:
    - High confidence for clear, specific questions (e.g., explicit error messages)
    - Low confidence for vague, ambiguous questions (e.g., "this doesn't work")
    - Medium confidence for questions with some context but missing details
    
    Returns a calibration score from 0.0 to 1.0:
    - 1.0 = perfectly calibrated (confidence matches expected range)
    - 0.0 = completely miscalibrated (confidence far from expected range)
    
    Args:
        inputs: The input data passed to the agent
        outputs: The outputs from the agent being evaluated
        reference_outputs: The expected/reference outputs from the dataset
        
    Returns:
        float: Calibration score between 0.0 and 1.0
    """
    try:
        # Get agent outputs
        overall_confidence = outputs.get("overall_confidence", 0.0)
        
        # Get expected confidence from reference
        expected_confidence = reference_outputs.get("expected_confidence", "medium")
        confidence_ranges = {
            "low": [0.0, 0.4],
            "medium": [0.4, 0.7], 
            "high": [0.7, 1.0]
        }
        
        # Convert label to range
        expected_range = confidence_ranges.get(expected_confidence, [0.0, 1.0])
        min_expected, max_expected = expected_range
        
        # Calculate calibration score
        if min_expected <= overall_confidence <= max_expected:
            # Perfect calibration
            calibration_score = 1.0
        else:
            # Penalize based on distance from expected range
            if overall_confidence < min_expected:
                distance = min_expected - overall_confidence
            else:  # overall_confidence > max_expected
                distance = overall_confidence - max_expected
            
            # Convert distance to penalty (0.0 to 1.0 scale)
            # Maximum penalty when distance >= 0.5 (e.g., expecting low 0.2 but got high 0.9)
            calibration_score = max(0.0, 1.0 - (distance * 2.0))
        
        return calibration_score
        
    except Exception as e:
        logger.error(f"Error in confidence calibration evaluator: {e}")
        return 0.0