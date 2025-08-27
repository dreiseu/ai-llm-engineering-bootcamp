from chainlit import run_app
import os
import sys

# Add the parent directory to the Python path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import *

# This creates the ASGI application that Vercel can serve
app = run_app("app.py", headless=True)