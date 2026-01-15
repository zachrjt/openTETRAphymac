import sys
import pytest
import numpy as np
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))

import src.tetraphymac.logical_channels as lc
import src.tetraphymac.physical_channels as bursts

np.random.seed(10)

###################################################################################################

def makeLogicalChannel(ChannelCLs, **kwargs):
    ch = ChannelCLs(**kwargs)
    inputData = ch.generateRndInput(1)
    ch.encodeType5Bits(inputData)
    return ch
    

@pytest.fixture
def makePhysicalChannel():
    # specific frequencies and main carrier, and channel number current don't affect behaviour
    def _mk(channel_type: bursts.PhyType):
        return bursts.Physical_Channel(
            channelNumber=1,
            mainCarrier=False,
            UL_Frequency=905.025,
            DL_Frequency=918.025,
            channelType=channel_type
        )
    
    return _mk

###################################################################################################

NORMAL_VALID = [
    #             name, phy, MN, FN, TN, bkn1_ctor, bkn2_ctor_or_none
    # test various traffic types
    ("tch72_only_tp_fn1", "TP", 1, 1, 1, lambda: makeLogicalChannel(lc.TCH_7_2), None),
    ("tch48_only_tp_fn1", "TP", 1, 1, 2, lambda: makeLogicalChannel(lc.TCH_4_8), None),
    ("tch24_only_tp_fn17", "TP", 50, 17, 3, lambda: makeLogicalChannel(lc.TCH_2_4), None),
    ("tchs_only_tp_fn2", "TP", 3, 2, 4, lambda: makeLogicalChannel(lc.TCH_S), None),

    ("schf_only_cp_fn1", "CP", 40, 1, 2, lambda: makeLogicalChannel(lc.SCH_F), None),
    ("schf_only_tp_fn18", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 3, lambda: makeLogicalChannel(lc.SCH_F), None),

    # STCH + TCH on TP, FN in [1..17]
    ("stch_tch_tp_fn1", "TP", 20, 17, 4, lambda: makeLogicalChannel(lc.STCH), lambda: makeLogicalChannel(lc.TCH_4_8)),
    ("stch_tch_tp_fn1", "TP", 4, 3, 2, lambda: makeLogicalChannel(lc.STCH), lambda: makeLogicalChannel(lc.TCH_7_2)),

    # SCH/HD + BNCH on CP or TP, choose MN/TN satisfying (MN+TN)%4==1 when FN==18
    ("schhd1_bnch_cp_fn18_mod1", "CP", 2, bursts.CONTROL_FRAME_NUMBER, 3, lambda: makeLogicalChannel(lc.SCH_HD), lambda: makeLogicalChannel(lc.BNCH)),
    ("schhd2_bnch_cp_fn18_mod1", "TP", 5, bursts.CONTROL_FRAME_NUMBER, 4, lambda: makeLogicalChannel(lc.SCH_HD), lambda: makeLogicalChannel(lc.BNCH)),

    # SCH/HD + BLCH on UP, any FN 1..18
    ("schhd_blch_up_fn5", "UP", 1, 5, 1, lambda: makeLogicalChannel(lc.SCH_HD), lambda: makeLogicalChannel(lc.BLCH)),
]

NORMAL_INVALID = [
    # name, phy, MN, FN, TN, bkn1_fn, bkn2_fn
    ("tch_on_cp_invalid", "CP", 1, 1, 1, lambda: makeLogicalChannel(lc.TCH_4_8, N=1), None),
    ("schf_on_tp_wrong_fn", "TP", 1, 1, 1, lambda: makeLogicalChannel(lc.SCH_F), None),
    ("stch_tch_on_cp_invalid", "CP", 1, 1, 1, lambda: makeLogicalChannel(lc.STCH), lambda: makeLogicalChannel(lc.TCH_4_8, N=1)),
    ("schhd_bnch_cp_fn18_modwrong", "CP", 1, bursts.CONTROL_FRAME_NUMBER, 1, lambda: makeLogicalChannel(lc.SCH_HD), lambda: makeLogicalChannel(lc.BNCH)),  # (1+1)%4=2
]

@pytest.mark.parametrize("name,phy,MN,FN,TN,bkn1_fn,bkn2_fn", NORMAL_VALID, ids=lambda x: x if isinstance(x, str) else None)
def test_normal_cont_downlink_valid(makePhysicalChannel, name, phy, MN, FN, TN, bkn1_fn, bkn2_fn):
    # test for continous
    phy_obj = makePhysicalChannel(phy)
    
    bkn1 = bkn1_fn()
    bbk  = makeLogicalChannel(lc.AACH)  # AACH always present
    bkn2 = None if bkn2_fn is None else bkn2_fn()

    burst = bursts.Normal_Cont_Downlink_Burst(phy_obj, MN, FN, TN)
    out = burst.constructBurstBitSequence(bkn1, bbk, bkn2, rampUpandDown=(False, False))
    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod


@pytest.mark.parametrize("name,phy,MN,FN,TN,bkn1_fn,bkn2_fn", NORMAL_VALID, ids=lambda x: x if isinstance(x, str) else None)
def test_normal_discont_downlink_valid(makePhysicalChannel, name, phy, MN, FN, TN, bkn1_fn, bkn2_fn):
    # test for continous
    phy_obj = makePhysicalChannel(phy)
    
    bkn1 = bkn1_fn()
    bbk  = makeLogicalChannel(lc.AACH)  # AACH always present
    bkn2 = None if bkn2_fn is None else bkn2_fn()

    burst = bursts.Normal_Discont_Downlink_Burst(phy_obj, MN, FN, TN)
    out = burst.constructBurstBitSequence(bkn1, bbk, bkn2, rampUpandDown=(False, False))
    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod


@pytest.mark.parametrize("name,phy,MN,FN,TN,bkn1_fn,bkn2_fn", NORMAL_INVALID, ids=lambda x: x if isinstance(x, str) else None)
def test_normal_cont_downlink_invalid(makePhysicalChannel, name, phy, MN, FN, TN, bkn1_fn, bkn2_fn):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Normal_Cont_Downlink_Burst(phy_obj, MN, FN, TN)

    bkn1 = bkn1_fn()
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = None if bkn2_fn is None else bkn2_fn()

    with pytest.raises(ValueError):
        burst.constructBurstBitSequence(bkn1, bbk, bkn2, rampUpandDown=(False, False))


@pytest.mark.parametrize("name,phy,MN,FN,TN,bkn1_fn,bkn2_fn", NORMAL_INVALID, ids=lambda x: x if isinstance(x, str) else None)
def test_normal_discont_downlink_invalid(makePhysicalChannel, name, phy, MN, FN, TN, bkn1_fn, bkn2_fn):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Normal_Discont_Downlink_Burst(phy_obj, MN, FN, TN)

    bkn1 = bkn1_fn()
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = None if bkn2_fn is None else bkn2_fn()

    with pytest.raises(ValueError):
        burst.constructBurstBitSequence(bkn1, bbk, bkn2, rampUpandDown=(False, False))
        

###################################################################################################

SYNC_VALID = [
    # BSCH + SCH_HD on TP/CP requires FN=18 and (MN+TN)%4==3
    ("bsch_schhd_tp_mod3", "TP", 2, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.SCH_HD),  # 2+1=3
    ("bsch_blch_cp_mod3",  "CP", 2, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.BLCH),
    # BSCH + BNCH must be on UP, FN in 1..18
    ("bsch_bnch_up_fn5",   "UP", 1, 5, 1, lc.BSCH, lc.BNCH),
]

SYNC_INVALID_BUILD = [
    # BSCH + SCH_HD: requires phy in {TP, CP}, FN=CONTROL_FRAME_NUMBER, and (MN+TN)%4==3
    ("bsch_schhd_invalid_phy_up", "UP", 2, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.SCH_HD),
    ("bsch_schhd_wrong_fn_not_18", "CP", 2, 1, 1, lc.BSCH, lc.SCH_HD),
    ("bsch_schhd_mod_rule_violation", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.SCH_HD),  # 1+1=2

    # BSCH + BLCH: same as BSCH + SCH_HD
    ("bsch_blch_invalid_phy_up", "UP", 2, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.BLCH),
    ("bsch_blch_wrong_fn_not_18", "TP", 2, 5, 1, lc.BSCH, lc.BLCH),
    ("bsch_blch_mod_rule_violation", "CP", 2, bursts.CONTROL_FRAME_NUMBER, 2, lc.BSCH, lc.BLCH),  # 2+2=0

    # BSCH + BNCH: must be on UP, FN in [1,18]
    ("bsch_bnch_invalid_phy_tp", "TP", 1, 5, 1, lc.BSCH, lc.BNCH),
    ("bsch_bnch_invalid_phy_cp", "CP", 1, 5, 1, lc.BSCH, lc.BNCH),

    # SCH_HD + BNCH: on TP requires FN=18 and (MN+TN)%4==1; on CP allows FN 1..18 with modulo checked when FN=18
    ("schhd_bnch_invalid_phy_up", "UP", 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.SCH_HD, lc.BNCH),
    ("schhd_bnch_tp_wrong_fn_not_18", "TP", 1, 1, 1, lc.SCH_HD, lc.BNCH),
    ("schhd_bnch_tp_mod_rule_violation", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.SCH_HD, lc.BNCH),  # 1+1=2 != 1
    ("schhd_bnch_cp_mod_rule_violation_on_fn18", "CP", 1, bursts.CONTROL_FRAME_NUMBER, 2, lc.SCH_HD, lc.BNCH),  # 1+2=3 != 1

    # SCH_HD + BLCH: on TP requires FN=18; on CP/UP requires FN in [1,18]
    ("schhd_blch_tp_wrong_fn_not_18", "TP", 1, 1, 1, lc.SCH_HD, lc.BLCH),
    
    # SCH_HD + SCH_HD: on TP requires FN=18; on CP/UP requires FN in [1,18]
    ("schhd_schhd_tp_wrong_fn_not_18", "TP", 1, 1, 1, lc.SCH_HD, lc.SCH_HD),
]

SYNC_INVALID_INIT = [
    # Phy invalid
    ("schhd_blch_invalid_phy_other", "XX", 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.SCH_HD, lc.BLCH),
    ("schhd_schhd_invalid_phy_other", "XX", 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.SCH_HD, lc.SCH_HD),

    # FN out of range
    ("init_fn_zero", "TP", 1, 0, 1, lc.BSCH, lc.SCH_HD),
    ("init_fn_19",   "CP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH + 1, 1, lc.BSCH, lc.SCH_HD),

    # TN out of range
    ("init_tn_zero", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 0, lc.BSCH, lc.SCH_HD),
    ("init_tn_5",    "TP", 1, bursts.CONTROL_FRAME_NUMBER, bursts.TDMAFRAME_TIMESLOT_LENGTH + 1, lc.BSCH, lc.SCH_HD),

    # MN out of range
    ("init_mn_zero", "TP", 0, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.SCH_HD),
    ("init_mn_61",   "TP", bursts.HYPERFRAME_MULTIFRAME_LENGTH + 1, bursts.CONTROL_FRAME_NUMBER, 1, lc.BSCH, lc.SCH_HD),
]

@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls", SYNC_VALID, ids=lambda x: x if isinstance(x, str) else None)
def test_sync_cont_downlink_valid(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Sync_Cont_Downlink_Burst(phy_obj, MN, FN, TN)

    sb   = makeLogicalChannel(SBcls)
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = makeLogicalChannel(BKN2cls)

    out = burst.constructBurstBitSequence(sb, bbk, bkn2, rampUpandDown=(False, False))

    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod

    # Synchronous downlink anchor: FREQ correction [14:94]
    assert np.array_equal(out[14:94], bursts.FREQUENCY_CORRECTION_FIELD)
    # Sync training region [214:252]
    assert np.array_equal(out[214:252], bursts.SYNCHRONIZATION_TRAINING_SEQUENCE)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls", SYNC_VALID, ids=lambda x: x if isinstance(x, str) else None)
def test_sync_discont_downlink_valid(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Sync_Discont_Downlink_Burst(phy_obj, MN, FN, TN)

    sb   = makeLogicalChannel(SBcls)
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = makeLogicalChannel(BKN2cls)

    out = burst.constructBurstBitSequence(sb, bbk, bkn2, rampUpandDown=(False, False))

    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod

    # Synchronous downlink anchor: FREQ correction [14:94]
    assert np.array_equal(out[14:94], bursts.FREQUENCY_CORRECTION_FIELD)
    # Sync training region [214:252]
    assert np.array_equal(out[214:252], bursts.SYNCHRONIZATION_TRAINING_SEQUENCE)


@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls",SYNC_INVALID_BUILD,ids=lambda x: x if isinstance(x, str) else None)
def test_sync_cont_downlink_invalid_build(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Sync_Cont_Downlink_Burst(phy_obj, MN, FN, TN)

    sb   = makeLogicalChannel(SBcls)
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = makeLogicalChannel(BKN2cls)

    with pytest.raises(ValueError):
        burst.constructBurstBitSequence(sb, bbk, bkn2, rampUpandDown=(False, False))

@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls",SYNC_INVALID_BUILD,ids=lambda x: x if isinstance(x, str) else None)
def test_sync_discont_downlink_invalid_build(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Sync_Discont_Downlink_Burst(phy_obj, MN, FN, TN)

    sb   = makeLogicalChannel(SBcls)
    bbk  = makeLogicalChannel(lc.AACH)
    bkn2 = makeLogicalChannel(BKN2cls)

    with pytest.raises(ValueError):
        burst.constructBurstBitSequence(sb, bbk, bkn2, rampUpandDown=(False, False))


@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls",SYNC_INVALID_INIT,ids=lambda x: x if isinstance(x, str) else None)
def test_sync_cont_downlink_invalid_init(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    with pytest.raises(ValueError):
        bursts.Sync_Cont_Downlink_Burst(phy_obj, MN, FN, TN)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SBcls,BKN2cls",SYNC_INVALID_INIT,ids=lambda x: x if isinstance(x, str) else None)
def test_sync_discont_downlink_invalid_init(makePhysicalChannel, name, phy, MN, FN, TN, SBcls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    with pytest.raises(ValueError):
        bursts.Sync_Discont_Downlink_Burst(phy_obj, MN, FN, TN)

###################################################################################################

CTRL_UL_VALID = [
    # On CP: FN can be any 1..18
    ("ctrl_ul_cp_fn1",  "CP", 1, 1,  1, 1),
    ("ctrl_ul_cp_fn9",  "CP", 1, 9,  2, 1),
    ("ctrl_ul_cp_fn18", "CP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH, 3, 1),

    # On TP: FN must be CONTROL_FRAME_NUMBER
    ("ctrl_ul_tp_fn18_ts1", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1),
    ("ctrl_ul_tp_fn18_ts4", "TP", 7, bursts.CONTROL_FRAME_NUMBER, bursts.TDMAFRAME_TIMESLOT_LENGTH, 1),

    ("ctrl_ul_cp_ssn2", "CP", 2, 5, 1, 2),
]

CTRL_UL_INVALID_INIT = [
    # Invalid phy for this burst type (UP is not allowed)
    ("ctrl_ul_invalid_phy_up", "UP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1),

    # TN out of range
    ("ctrl_ul_invalid_tn0", "CP", 1, 1, 0, 1),
    ("ctrl_ul_invalid_tn_hi", "CP", 1, 1, bursts.TDMAFRAME_TIMESLOT_LENGTH + 1, 1),

    # MN out of range
    ("ctrl_ul_invalid_mn0", "CP", 0, 1, 1, 1),
    ("ctrl_ul_invalid_mn_hi", "CP", bursts.HYPERFRAME_MULTIFRAME_LENGTH + 1, 1, 1, 1),

    # FN out of range
    ("ctrl_ul_invalid_fn0", "CP", 1, 0, 1, 1),
    ("ctrl_ul_invalid_fn_hi", "CP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH + 1, 1, 1),

    # SSN out of range
    ("ctrl_ul_invalid_ssn0", "CP", 1, 1, 1, 0),
    ("ctrl_ul_invalid_ssn_hi", "CP", 1, 1, 1, bursts.TIMESLOT_SUBSLOT_LENGTH + 1),
]

CTRL_UL_INVALID_BUILD = [
    # On TP, FN must be CONTROL_FRAME_NUMBER (usually 18). Choose any FN in-range but not 18.
    ("ctrl_ul_tp_wrong_fn1", "TP", 1, 1, 1, 1),
    ("ctrl_ul_tp_wrong_fn17", "TP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH - 1, 1, 1),

    ("ctrl_ul_wrong_logical_channel", "CP", 1, 1, 1, 1),  # and pass lc.SCH_HD instead of lc.SCH_HU
]


@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN",CTRL_UL_VALID,ids=lambda x: x if isinstance(x, str) else None)
def test_control_uplink_valid(makePhysicalChannel, name, phy, MN, FN, TN, SSN):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Control_Uplink(phy_obj, MN, FN, TN, SSN)

    sch_hu = makeLogicalChannel(lc.SCH_HU)  # must produce type5Blocks[0] length >= 168
    out = burst.constructBurstBitSequence(sch_hu)

    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod
    # quick structural anchors
    d = burst.startGuardBitPeriod
    assert np.array_equal(out[d:d+4], bursts.TAIL_BITS)
    assert np.array_equal(out[d+88:d+118], bursts.EXTENDED_TRAINING_SEQUENCE)
    assert np.array_equal(out[d+202:d+206], bursts.TAIL_BITS)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN",CTRL_UL_INVALID_INIT,ids=lambda x: x if isinstance(x, str) else None)
def test_control_uplink_invalid_init(makePhysicalChannel, name, phy, MN, FN, TN, SSN):
    phy_obj = makePhysicalChannel(phy)
    with pytest.raises(ValueError):
        bursts.Control_Uplink(phy_obj, MN, FN, TN, SSN)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN",CTRL_UL_INVALID_BUILD,ids=lambda x: x if isinstance(x, str) else None)
def test_control_uplink_invalid_build(makePhysicalChannel, name, phy, MN, FN, TN, SSN):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Control_Uplink(phy_obj, MN, FN, TN, SSN)

    if name == "ctrl_ul_wrong_logical_channel":
        sch_hd = makeLogicalChannel(lc.SCH_HD)
        with pytest.raises(ValueError):
            burst.constructBurstBitSequence(sch_hd) #type: ignore
    else:
        sch_hu = makeLogicalChannel(lc.SCH_HU)
        with pytest.raises(ValueError):
            burst.constructBurstBitSequence(sch_hu)

###################################################################################################

NORM_UL_VALID = [
    # Pure traffic: (TRAFFIC, None) requires TP, FN 1..17
    ("ul_pure_tch_tp_fn1",  "TP", 1, 1,  1, 1, lc.TCH_4_8, None),
    ("ul_pure_tch_tp_fn17", "TP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH - 1, 2, 1, lc.TCH_2_4, None),

    # Pure control SCH/F: (CONTROL, None)
    # On CP: FN 1..18
    ("ul_schf_cp_fn1",  "CP", 1, 1,  1, 1, lc.SCH_F, None),
    ("ul_schf_cp_fn18", "CP", 2, bursts.CONTROL_FRAME_NUMBER, 2, 1, lc.SCH_F, None),
    # On TP: FN must be 18
    ("ul_schf_tp_fn18", "TP", 3, bursts.CONTROL_FRAME_NUMBER, 3, 1, lc.SCH_F, None),

    # (CONTROL, TRAFFIC) on TP FN 1..17 
    ("ul_stch_tch_tp_fn1",  "TP", 1, 1,  1, 1, lc.STCH, lc.TCH_4_8),
    ("ul_stch_tch_tp_fn17", "TP", 2, bursts.MULTIFRAME_TDMAFRAME_LENGTH - 1, 2, 1, lc.STCH, lc.TCH_2_4),

    # (CONTROL, CONTROL) on TP FN 1..17
    ("ul_stch_stch_tp_fn5", "TP", 5, 5,  3, 1, lc.STCH, lc.STCH),

    # SSN of 2
    ("ul_pure_tch_tp_ssn2", "TP", 1, 1, 1, 2, lc.TCH_4_8, None),
]

NORM_UL_INVALID_INIT = [
    # invalid phy type (UP not allowed)
    ("ul_invalid_phy_up", "UP", 1, 1, 1, 1, lc.TCH_4_8, None),

    # FN out of range (now validated in base init)
    ("ul_invalid_fn0",  "TP", 1, 0, 1, 1, lc.TCH_4_8, None),
    ("ul_invalid_fn19", "CP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH + 1, 1, 1, lc.SCH_F, None),

    # TN out of range
    ("ul_invalid_tn0",  "TP", 1, 1, 0, 1, lc.TCH_4_8, None),
    ("ul_invalid_tn_hi","TP", 1, 1, bursts.TDMAFRAME_TIMESLOT_LENGTH + 1, 1, lc.TCH_4_8, None),

    # MN out of range
    ("ul_invalid_mn0",  "TP", 0, 1, 1, 1, lc.TCH_4_8, None),
    ("ul_invalid_mn_hi","TP", bursts.HYPERFRAME_MULTIFRAME_LENGTH + 1, 1, 1, 1, lc.TCH_4_8, None),

    # SSN out of range
    ("ul_invalid_ssn0",  "TP", 1, 1, 1, 0, lc.TCH_4_8, None),
    ("ul_invalid_ssn_hi","TP", 1, 1, 1, bursts.TIMESLOT_SUBSLOT_LENGTH + 1, lc.TCH_4_8, None),
]

NORM_UL_INVALID_BUILD = [
    # Pure traffic on wrong phy 
    ("ul_pure_tch_on_cp", "CP", 1, 1, 1, 1, lc.TCH_4_8, None),  # expects TP

    # Pure traffic wrong FN (FN must be 1..17) 
    ("ul_pure_tch_tp_fn18", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.TCH_4_8, None),

    # SCH/F on TP wrong FN (must be 18) 
    ("ul_schf_tp_fn1", "TP", 1, 1, 1, 1, lc.SCH_F, None),

    # (CONTROL, TRAFFIC) on wrong phy (must be TP) 
    ("ul_stch_tch_on_cp", "CP", 1, 1, 1, 1, lc.STCH, lc.TCH_4_8),

    #(CONTROL, TRAFFIC) wrong FN (must be 1..17) 
    ("ul_stch_tch_tp_fn18", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.STCH, lc.TCH_4_8),

    #(CONTROL, CONTROL) wrong FN (must be 1..17) 
    ("ul_stch_stch_tp_fn18", "TP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.STCH, lc.STCH),

    # bkn1=TRAFFIC, bkn2=TRAFFIC => falls to default case -> ValueError
    ("ul_tch_tch_invalid_combo", "TP", 1, 1, 1, 1, lc.TCH_4_8, lc.TCH_2_4),

    # bkn1=TRAFFIC and bkn2=CONTROL (STCH) is not matched 
    ("ul_tch_stch_invalid_combo", "TP", 1, 1, 1, 1, lc.TCH_4_8, lc.STCH),

    # bkn1=SCH_F(control) but bkn2=STCH(control) hits (CONTROL, CONTROL) branch.
    ("ul_schf_stch_should_be_invalid", "TP", 1, 1, 1, 1, lc.SCH_F, lc.STCH),
]

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,BKN1cls,BKN2cls",NORM_UL_VALID,ids=lambda x: x if isinstance(x, str) else None)
def test_normal_uplink_valid(makePhysicalChannel, name, phy, MN, FN, TN, SSN, BKN1cls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Normal_Uplink_Burst(phy_obj, MN, FN, TN, SSN)

    bkn1 = makeLogicalChannel(BKN1cls)
    if BKN2cls is None:
        bkn2 = None
    else:
        bkn2 = makeLogicalChannel(BKN2cls)

    out = burst.constructBurstBitSequence(bkn1, bkn2, rampUpandDown=(True, True))
    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod

    d = burst.startGuardBitPeriod
    assert np.array_equal(out[d:d+4], bursts.TAIL_BITS)
    assert np.array_equal(out[d+220:d+242], bursts.NORMAL_TRAINING_SEQUENCE[0]) or np.array_equal(out[d+220:d+242], bursts.NORMAL_TRAINING_SEQUENCE[1])
    assert np.array_equal(out[d+458:d+462], bursts.TAIL_BITS)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,BKN1cls,BKN2cls",NORM_UL_INVALID_INIT,ids=lambda x: x if isinstance(x, str) else None)
def test_normal_uplink_invalid_init(makePhysicalChannel, name, phy, MN, FN, TN, SSN, BKN1cls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    with pytest.raises(ValueError):
        bursts.Normal_Uplink_Burst(phy_obj, MN, FN, TN, SSN)

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,BKN1cls,BKN2cls",NORM_UL_INVALID_BUILD,ids=lambda x: x if isinstance(x, str) else None)
def test_normal_uplink_invalid_build(makePhysicalChannel, name, phy, MN, FN, TN, SSN, BKN1cls, BKN2cls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Normal_Uplink_Burst(phy_obj, MN, FN, TN, SSN)
    
    bkn1 = makeLogicalChannel(BKN1cls)
    if BKN2cls is None:
        bkn2 = None
    else:
        bkn2 = makeLogicalChannel(BKN2cls)

    with pytest.raises(ValueError):
        out = burst.constructBurstBitSequence(bkn1, bkn2, rampUpandDown=(True, True))

###################################################################################################

LIN_UL_VALID = [
    # On CP/TP: must be control frame and (MN+TN)%4 == 3
    ("lin_ul_cp_mn2_tn1_mod3", "CP", 2, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.CLCH),  # 2+1=3
    ("lin_ul_tp_mn3_tn4_mod3", "TP", 3, bursts.CONTROL_FRAME_NUMBER, 4, 1, lc.CLCH),  # 3+4=7 -> 3 mod 4

    # On UP: no FN/modulo; only requires CLCH channel
    ("lin_ul_up_any_fn1",      "UP", 1, 1, 1, 1, lc.CLCH),
    ("lin_ul_up_fn18_ok",      "UP", 5, bursts.CONTROL_FRAME_NUMBER, 2, 1, lc.CLCH),
]

LIN_UL_INVALID_INIT = [
    # invalid TN
    ("lin_ul_init_tn0",    "CP", 1, bursts.CONTROL_FRAME_NUMBER, 0, 1, lc.CLCH),
    ("lin_ul_init_tn_hi",  "CP", 1, bursts.CONTROL_FRAME_NUMBER, bursts.TDMAFRAME_TIMESLOT_LENGTH + 1, 1, lc.CLCH),

    # invalid MN
    ("lin_ul_init_mn0",    "CP", 0, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.CLCH),
    ("lin_ul_init_mn_hi",  "CP", bursts.HYPERFRAME_MULTIFRAME_LENGTH + 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.CLCH),

    # invalid FN 
    ("lin_ul_init_fn0",    "CP", 1, 0, 1, 1, lc.CLCH),
    ("lin_ul_init_fn_hi",  "CP", 1, bursts.MULTIFRAME_TDMAFRAME_LENGTH + 1, 1, 1, lc.CLCH),

    # invalid SSN
    ("lin_ul_init_ssn0",   "CP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 0, lc.CLCH),
    ("lin_ul_init_ssn_hi", "CP", 1, bursts.CONTROL_FRAME_NUMBER, 1, bursts.TIMESLOT_SUBSLOT_LENGTH + 1, lc.CLCH),

    # invalid phy string
    ("lin_ul_init_phy_invalid", "XX", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.CLCH),
]

LIN_UL_INVALID_BUILD = [
    #  Wrong FN on CP/TP: must be CONTROL_FRAME_NUMBER 
    ("lin_ul_cp_wrong_fn1",  "CP", 2, 1, 1, 1, lc.CLCH),
    ("lin_ul_tp_wrong_fn17", "TP", 2, bursts.MULTIFRAME_TDMAFRAME_LENGTH - 1, 1, 1, lc.CLCH),

    #  Modulo rule on CP/TP: (MN+TN)%4 must be 3 
    ("lin_ul_cp_mod_bad",    "CP", 1, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.CLCH),  # 1+1=2
    ("lin_ul_tp_mod_bad",    "TP", 2, bursts.CONTROL_FRAME_NUMBER, 2, 1, lc.CLCH),  # 2+2=0

    #  Wrong logical channel: must be CLCH 
    # Pick any non-CLCH logical channel class that has .channel != CLCH_CHANNEL
    ("lin_ul_cp_wrong_lc_aach", "CP", 2, bursts.CONTROL_FRAME_NUMBER, 1, 1, lc.AACH),
    ("lin_ul_up_wrong_lc_schf", "UP", 1, 1, 1, 1, lc.SCH_F),

    # Valid mod, but invalid subslot
    ("lin_ul_cp_ssn2_mod3",    "CP", 6, bursts.CONTROL_FRAME_NUMBER, 1, 2, lc.CLCH),  # 6+1=7 -> 3 mod 4
]

@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,LCcls",LIN_UL_VALID,ids=lambda x: x if isinstance(x, str) else None)
def test_linearization_uplink_valid(makePhysicalChannel, name, phy, MN, FN, TN, SSN, LCcls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Linearization_Uplink_Burst(phy_obj, MN, FN, TN, SSN)

    ch = makeLogicalChannel(LCcls)
    out = burst.constructBurstBitSequence(ch)

    assert len(out) == (burst.SNmax * 2) + burst.startGuardBitPeriod + burst.endGuardBitPeriod

    d = burst.startGuardBitPeriod
    # payload region should be exactly the CLCH bits
    assert np.array_equal(out[d:d+238], np.array(ch.type5Blocks[0][:238], dtype=np.uint8))


@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,LCcls",LIN_UL_INVALID_INIT,ids=lambda x: x if isinstance(x, str) else None)
def test_linearization_uplink_invalid_init(makePhysicalChannel, name, phy, MN, FN, TN, SSN, LCcls):
    phy_obj = makePhysicalChannel(phy)
    with pytest.raises(ValueError):
        bursts.Linearization_Uplink_Burst(phy_obj, MN, FN, TN, SSN)


@pytest.mark.parametrize("name,phy,MN,FN,TN,SSN,LCcls",LIN_UL_INVALID_BUILD,ids=lambda x: x if isinstance(x, str) else None)
def test_linearization_uplink_invalid_build(makePhysicalChannel, name, phy, MN, FN, TN, SSN, LCcls):
    phy_obj = makePhysicalChannel(phy)
    burst = bursts.Linearization_Uplink_Burst(phy_obj, MN, FN, TN, SSN)

    ch = makeLogicalChannel(LCcls)

    with pytest.raises(ValueError):
        burst.constructBurstBitSequence(ch)
