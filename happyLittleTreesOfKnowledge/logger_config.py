import logging
import os


def setup_logger(name=None, level=None):
    """
    Set up a configured logger for the application.
    
    Args:
        name: Logger name. If None, uses the calling module's name
        level: Logging level. If None, uses INFO or environment variable LOG_LEVEL
    
    Returns:
        Configured logger instance
    """
    if name is None:
        name = __name__
    
    if level is None:
        level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level
        logger.setLevel(level)
    
    return logger