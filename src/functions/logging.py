import json
import os



def setup_logging(log_file: str) -> None:
    """
    Set up logging configuration to output logs in JSON format to the specified file
    
    :param log_file: The file to log events to
    :return: None
    """
    
    # If the log file already exists, it will be deleted to start fresh
    if os.path.exists(log_file):
        os.remove(log_file)  
        
    # if the log file does not exist, it will be created
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    pass

def log_event(event: dict, log_file: str) -> None:
    """
    Log an event as a JSON string
    
    :param event: The event to log
    :param log_file: The file to log the event to
    :return: None
    """
    
    # Save the event as a JSON string to the log file
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")
