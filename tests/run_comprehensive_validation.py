#!/usr/bin/env python3
"""
Comprehensive multimodal validation runner for Korean-English translation system.
Executes validation tests for text, images, and audio with detailed reporting.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Run comprehensive multimodal validation."""
    print("🚀 Starting comprehensive multimodal validation...")

    try:
        from validation.multimodal_validator import main as validate_main

        validate_main()
        print("✅ Comprehensive validation completed successfully!")

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
