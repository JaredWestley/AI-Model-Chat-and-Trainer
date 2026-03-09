"""
CLI chat entry point.
Usage: python chat_cli.py
       python chat_cli.py --checkpoint checkpoints/ckpt_best.pt
       python chat_cli.py --device cpu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.chat import main

if __name__ == "__main__":
    main()
