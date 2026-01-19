"""
RunPod Handler Entry Point

This file re-exports the main handler for RunPod Hub compatibility.
The actual implementation is in handler.py at the root.
"""

import runpod
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main handler
from handler import handler

# RunPod Serverless entry point
runpod.serverless.start({"handler": handler})
