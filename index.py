import sys
import os

# Make project root importable so app.py, models, and data files resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app
