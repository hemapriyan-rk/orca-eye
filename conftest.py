"""
pytest configuration for ORCA EYE tests.
Adds the project root to sys.path so test files can import orca modules directly.
"""
import sys
from pathlib import Path

# Add project root (d:/orca) to path
sys.path.insert(0, str(Path(__file__).parent))
