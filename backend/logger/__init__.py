import logging
import os

from from_root import from_root
from datetime import datetime


backend_root = f"{from_root()}/backend"

project_root = os.getenv("PROJECT_ROOT", backend_root)

log_dir = 'app_data/logs'

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(project_root, log_dir, LOG_FILE)

os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)