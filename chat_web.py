"""
Gradio web UI entry point.
Usage: python chat_web.py
       python chat_web.py --port 8080
       python chat_web.py --share   (creates public URL)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.web_ui import main

if __name__ == "__main__":
    main()
