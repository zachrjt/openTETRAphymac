import sys
import pytest
import numpy as np
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))

import src.tetraphymac.logical_channels as lc

np.random.seed(10)

# parameterize the arguments so we only need to pass keyword arguments which are already formatted in a dict
CASES = [
    # Traffic channels with N
    (lc.TCH_4_8, {"N": 8}, 16),
    (lc.TCH_4_8, {"N": 8}, 4),
    (lc.TCH_4_8, {"N": 4}, 2),
    (lc.TCH_4_8, {"N": 1}, 2),
    (lc.TCH_4_8, {"N": 1}, 1),

    (lc.TCH_2_4, {"N": 8}, 16),
    (lc.TCH_2_4, {"N": 8}, 4),
    (lc.TCH_2_4, {"N": 4}, 2),
    (lc.TCH_2_4, {"N": 1}, 2),
    (lc.TCH_2_4, {"N": 1}, 1),

    (lc.TCH_7_2, {"N": 1}, 8),
    (lc.TCH_7_2, {"N": 1}, 1),

    # Traffic channels with slotLength
    (lc.TCH_S, {"N": 1, "slotLength": "full"}, 1),
    (lc.TCH_S, {"N": 1, "slotLength": "half"}, 1),

    # Control channels: no args
    (lc.BNCH, {}, 1),
    (lc.BSCH, {}, 1),
    (lc.SCH_F, {}, 1),
    (lc.SCH_HD, {}, 1),
    (lc.SCH_HU, {}, 1),
    (lc.AACH, {}, 1),
    (lc.STCH, {}, 1),
]

def _assignArguments(case):
    ChannelCls, kwargs, M = case
    kw = ",".join(f"{k}={v}" for k,v in kwargs.items()) or "noargs"
    return f"{ChannelCls.__name__}[{kw}]-M{M}"

@pytest.mark.parametrize(
    "ChannelCls,init_kwargs,M", 
    [pytest.param(*case, id=_assignArguments(case)) for case in CASES]
)

def test_traffic_channels(ChannelCls, init_kwargs, M):
    # pass dict containing keyword-arguments, must use ** operator to do that
    tx = ChannelCls(**init_kwargs)
    tx.encodeType5Bits([tx.generateRndInput() for _ in range(M)])

    rx = ChannelCls(**init_kwargs)
    rx.decodeType5Bits(tx.type5Blocks)
    assert rx.type1Blocks == tx.type1Blocks