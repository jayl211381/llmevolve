import logging

"""
Logger module for evolutionary llm
"""
LOGGER = logging.getLogger("ea_llm")
LOGGER.setLevel(logging.DEBUG)

# Create console handler if not already added
if not LOGGER.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    LOGGER.addHandler(console_handler)

