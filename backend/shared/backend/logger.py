import logging
import sys
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line_number": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(JSONFormatter())
        logger.addHandler(c_handler)
        
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(JSONFormatter())
        logger.addHandler(f_handler)
        
    return logger
