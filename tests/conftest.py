"""Test configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to path for imports (allows `from src.xxx import xxx`)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
