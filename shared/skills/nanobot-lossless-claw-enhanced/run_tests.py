#!/usr/bin/env python3
"""Simple test runner for LCM modules.

Run tests without pytest dependency.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.test_estimate_tokens import TestEstimateTokens
from tests.test_session_patterns import TestSessionPatterns
from tests.test_fts5_sanitize import TestFts5Sanitize
from tests.test_full_text_fallback import TestFullTextFallback
from tests.test_transcript_repair import TestTranscriptRepair


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    loader.discoverTests(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(loader)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    print(f"\nAll tests passed: {success}")
    sys.exit(0 if success else 1)
