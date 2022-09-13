"""
Predicting Purchase Intention model package params
load and validate the environment variables in the `.env`
"""

import os

LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))

SOURCE = os.path.expanduser(os.environ.get("SOURCE"))
