"""Paths of the project."""
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")  # raw downloads
SETUP_DATA_PATH = os.path.join(ROOT_PATH, "setup_data")  # extracted copies used by the handlers
