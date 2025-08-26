"""
Simple test to isolate the hanging issue
"""

def test_basic_imports():
    print("Testing basic imports...")
    
    try:
        import uuid
        print("✓ uuid import OK")
        
        import time
        print("✓ time import OK")
        
        from datetime import datetime
        print("✓ datetime import OK")
        
        from enum import Enum
        print("✓ enum import OK")
        
        from typing import Dict, Any, Optional, List
        print("✓ typing import OK")
        
        from dataclasses import dataclass, field
        print("✓ dataclasses import OK")
        
        from collections import deque
        print("✓ collections import OK")
        
        import heapq
        print("✓ heapq import OK")
        
        import threading
        print("✓ threading import OK")
        
        import random
        print("✓ random import OK")
        
        import math
        print("✓ math import OK")
        
        import numpy as np
        print("✓ numpy import OK")
        
        from scipy import stats
        print("✓ scipy.stats import OK")
        
        print("All imports successful!")
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_imports()
