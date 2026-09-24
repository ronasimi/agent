#!/usr/bin/env python3
"""Print the P2 browser/UI regression dashboard as JSON."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.browser_benchmark import regression_dashboard
print(json.dumps(regression_dashboard(), indent=2, sort_keys=True))
