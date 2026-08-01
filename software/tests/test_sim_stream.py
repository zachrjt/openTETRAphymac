# type: ignore
# pylint: skip-file
# flake8: noqa
import sys
import pytest
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
import src.tetraphymac.constants as tetraConstants
import src.tetraphymac.ch_simulator as tetraCh


