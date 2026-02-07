# pylint: skip-file
# flake8: noqa
# type: ignore
import sys
import pytest
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
import numpy as np

import src.tetraphymac.logical_channels as lc

np.random.seed(10)

# parameterize the arguments so we only need to pass keyword arguments which are already formatted in a dict
CASES = [
    # Traffic channels with n
    (lc.TCH_4_8, {"n": 8}, 16),
    (lc.TCH_4_8, {"n": 8}, 4),
    (lc.TCH_4_8, {"n": 4}, 2),
    (lc.TCH_4_8, {"n": 1}, 2),
    (lc.TCH_4_8, {"n": 1}, 1),

    (lc.TCH_2_4, {"n": 8}, 16),
    (lc.TCH_2_4, {"n": 8}, 4),
    (lc.TCH_2_4, {"n": 4}, 2),
    (lc.TCH_2_4, {"n": 1}, 2),
    (lc.TCH_2_4, {"n": 1}, 1),

    (lc.TCH_7_2, {"n": 1}, 8),
    (lc.TCH_7_2, {"n": 1}, 1),

    # Traffic channels with slot_length
    (lc.TCH_S, {"n": 1, "slot_length": "full"}, 1),
    (lc.TCH_S, {"n": 1, "slot_length": "half"}, 1),

    # Control channels: no args
    (lc.BNCH, {}, 1),
    (lc.BSCH, {}, 1),
    (lc.SCH_F, {}, 1),
    (lc.SCH_HD, {}, 1),
    (lc.SCH_HU, {}, 1),
    (lc.AACH, {}, 1),
    (lc.STCH, {}, 1),

    # Linearization channels
    (lc.BLCH, {}, 1),
    (lc.CLCH, {}, 1),
]

CASES_SPEECH = [
    # Traffic channels with slot_length
    (lc.TCH_S, {"n": 1, "slot_length": "full"}, 1),
]

def _assignArguments(case):
    ChannelCls, kwargs, M = case
    kw = ",".join(f"{k}={v}" for k,v in kwargs.items()) or "noargs"
    return f"{ChannelCls.__name__}[{kw}]-M{M}"

@pytest.mark.parametrize(
    "ChannelCls,init_kwargs,M", 
    [pytest.param(*case, id=_assignArguments(case)) for case in CASES]
)

def test_traffic_channels(ChannelCls, init_kwargs, M:int):
    # pass dict containing keyword-arguments, must use ** operator to do that
    tx = ChannelCls(**init_kwargs)
    inputData = tx.generate_rnd_input(M)
    tx.encode_type5_bits(inputData)

    rx = ChannelCls(**init_kwargs)
    rx.decode_type5_bits(tx.type_5_blocks)
    assert (rx.type_1_blocks == tx.type_1_blocks).all()

# Test dedicated to verifying the speech traffic logical channel, can steal data on the fly with its' method
@pytest.mark.parametrize("ChannelCls,init_kwargs,M", [pytest.param(*case, id=_assignArguments(case)) for case in CASES_SPEECH])
def test_speech_slots(ChannelCls, init_kwargs, M:int):
    # pass dict containing keyword-arguments, must use ** operator to do that
    tx = ChannelCls(**init_kwargs)
    inputData = tx.generate_rnd_input(M)
    tx.encode_type5_bits(inputData)

    rx = ChannelCls(**init_kwargs)
    rx.decode_type5_bits(tx.type_5_blocks)
    assert (rx.type_1_blocks == tx.type_1_blocks).all()

    tx.steal_block_a()
    
    rx = ChannelCls(n=1, slot_length="half")
    rx.decode_type5_bits(tx.type_5_blocks)
    assert (rx.type_1_blocks == tx.type_1_blocks).all()