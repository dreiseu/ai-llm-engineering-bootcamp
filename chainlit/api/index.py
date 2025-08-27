import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import app
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Import chainlit after setting up the path
import chainlit as cl
from chainlit.server import app as chainlit_app

# Import our app module to register the handlers
import app

# Export the ASGI application for Vercel
app = chainlit_app