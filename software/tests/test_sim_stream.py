# type: ignore
# pylint: skip-file
# flake8: noqa
import sys
import pytest
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
import src.tetraphymac.logical_channels as tetraLch
import src.tetraphymac.physical_channels as tetraPhy
import src.tetraphymac.transmitter as tetraTx
import src.tetraphymac.tx_rx_utilities as tetraUtil
import src.tetraphymac.constants as tetraConstants
import src.tetraphymac.measurements as tetraMeas
import src.tetraphymac.simulation_streaming as tetraStream

from src.tetraphymac.constants import StreamPosition as sp



def get_default_burst_channels(burst_type, length):
    if burst_type  == tetraPhy.NormalUplinkBurst:
        rtrn_log_list = ([tetraLch.TCH_7_2() for _ in range(length)],)
    elif burst_type == tetraPhy.ControlUplink:
        rtrn_log_list = ([tetraLch.SCH_HU() for _ in range(length)],)
    elif burst_type == tetraPhy.LinearizationUplinkBurst:
        rtrn_log_list = ([tetraLch.CLCH() for _ in range(length)],)
    elif burst_type == tetraPhy.SyncContDownlinkBurst or burst_type == tetraPhy.SyncDiscontDownlinkBurst:
        rtrn_log_list = ([tetraLch.BSCH() for _ in range(length)], [tetraLch.AACH() for _ in range(length)],
                         [tetraLch.SCH_HD() for _ in range(length)])
    elif burst_type == tetraPhy.NormalContDownlinkBurst or burst_type == tetraPhy.NormalDiscontDownlinkBurst:
        rtrn_log_list = ([tetraLch.SCH_F() for _ in range(length)], [tetraLch.AACH() for _ in range(length)])
    else:
        raise ValueError(f"Burst type passed to `get_burst_channels` is: {burst_type}, unexpected.")

    return rtrn_log_list

def schedule_wrap(streamer: tetraStream.BurstStreamBuilder, burst_type, channels, *,
                  allow_ms_adjacent_slot_ramp_bypass=False,
                  bs_timeshare_adj_slot_ramp_bypass=False,
                  continuous_with_prior_blocks=True,
                  forced_scheduling=True, fill_empty_channels=True):
    streamer.schedule_bursts(burst_type=burst_type,
                            input_logical_ch=channels,
                            ms_adj_slot_ramp_bypass=allow_ms_adjacent_slot_ramp_bypass,
                            bs_timeshare_adj_slot_ramp_bypass=bs_timeshare_adj_slot_ramp_bypass,
                            continuous_with_prior_blocks=continuous_with_prior_blocks,
                            forced_scheduling=forced_scheduling,
                            fill_empty_channels=fill_empty_channels)

# Test verifyies that the streamer will fill empty logical channels
def test_empty_fill_valid():
    # Verify that fill_empty_channels argument works correctly
    tetra_streamer = tetraStream.BurstStreamBuilder()
    pkt_traffic_ch = tetraLch.TCH_4_8(n=4)
    tetra_streamer.schedule_bursts(burst_type=tetraPhy.NormalUplinkBurst,
                                    input_logical_ch=([pkt_traffic_ch],),
                                    ms_adj_slot_ramp_bypass=False,
                                    continuous_with_prior_blocks=False,
                                    forced_scheduling=True,
                                    fill_empty_channels=True)

    burst_list = tetra_streamer.get_scheduled_bursts(return_all_future_bursts=True,
                                                    increment_time_on_call=False)
    for burst in burst_list:
        index = burst.logical_ch_block_indices[0]
        assert burst.logical_channels[0].type_5_blocks[index].size > 0

    pkt_traffic_ch2 = tetraLch.TCH_4_8(n=4)

    with pytest.raises(ValueError):
        tetra_streamer.schedule_bursts(burst_type=tetraPhy.NormalUplinkBurst,
                                        input_logical_ch=([pkt_traffic_ch2],),
                                        ms_adj_slot_ramp_bypass=False,
                                        continuous_with_prior_blocks=False,
                                        forced_scheduling=True,
                                        fill_empty_channels=False)

# test verifies that the streamer will not fill empty logical channels
def test_empty_fill_invalid():
    tetra_streamer = tetraStream.BurstStreamBuilder()
    pkt_traffic_ch = tetraLch.TCH_4_8(n=4)
    pkt_traffic_ch2 = tetraLch.TCH_4_8(n=4)
    
    with pytest.raises(ValueError):
        tetra_streamer.schedule_bursts(burst_type=tetraPhy.NormalUplinkBurst,
                                        input_logical_ch=([pkt_traffic_ch2],),
                                        ms_adj_slot_ramp_bypass=False,
                                        continuous_with_prior_blocks=False,
                                        forced_scheduling=True,
                                        fill_empty_channels=False)

CONT_PRIOR_CASES = [
    # Normal uplink
    ("normal_ul_cont_prior_w_iso_normal_ul", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_cont_normal_ul", tetraPhy.NormalUplinkBurst, 2, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_iso_control_ul", tetraPhy.ControlUplink, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_iso_linear_ul", tetraPhy.LinearizationUplinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_cont_sync_dl", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_cont_normal_dl", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_discon_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_cont_prior_w_discon_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_without_cont_prior", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (True, True, False),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("normal_ul_without_ramp_bypss", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (False, True, True),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),
    ("normal_ul_with_neither", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalUplinkBurst, 3, (False, True, False),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),

    # Discont. Normal Downlink
    ("discon_normal_dl_cont_prior_w_iso_discon_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_cont_discon_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 2, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_iso_discon_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_iso_control_ul", tetraPhy.ControlUplink, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_iso_linear_ul", tetraPhy.LinearizationUplinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_cont_sync_dl", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_cont_normal_dl", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_cont_prior_w_cont_normal_ul", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_without_cont_prior", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, True, False),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_normal_dl_without_ramp_bypss", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, False, True),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),
    ("discon_normal_dl_with_neither", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalDiscontDownlinkBurst, 3, (True, False, False),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),

    # Discont. Sync Downlink
    ("discon_sync_dl_cont_prior_w_iso_discon_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_cont_discon_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 2, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_iso_discon_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_iso_control_ul", tetraPhy.ControlUplink, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_iso_linear_ul", tetraPhy.LinearizationUplinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_cont_sync_dl", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_cont_normal_dl", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_cont_prior_w_cont_normal_ul", tetraPhy.NormalUplinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_without_cont_prior", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, True, False),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("discon_sync_dl_without_ramp_bypss", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, False, True),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),
    ("discon_sync_dl_with_neither", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.SyncDiscontDownlinkBurst, 3, (True, False, False),
        (sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST, sp.ISOLATED_BURST)),
    
    # Cont. Normal Downlink
    ("con_normal_dl_cont_prior_w_iso_con_normal_dl", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_cont_con_normal_dl", tetraPhy.NormalContDownlinkBurst, 2, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_iso_con_sync_dl", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_iso_control_ul", tetraPhy.ControlUplink, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_iso_linear_ul", tetraPhy.LinearizationUplinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_cont_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_cont_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_cont_prior_w_cont_normal_ul", tetraPhy.NormalUplinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_without_cont_prior", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, True, False),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_without_ramp_bypss", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (True, False, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_normal_dl_with_neither", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.NormalContDownlinkBurst, 3, (False, False, True),

        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    # Cont. Sync Downlink
    ("con_sync_dl_cont_prior_w_iso_con_sync_dl", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_cont_con_sync_dl", tetraPhy.SyncContDownlinkBurst, 2, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_iso_normal_sync_dl", tetraPhy.NormalContDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_iso_control_ul", tetraPhy.ControlUplink, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_iso_linear_ul", tetraPhy.LinearizationUplinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_cont_sync_dl", tetraPhy.SyncDiscontDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_cont_normal_dl", tetraPhy.NormalDiscontDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_cont_prior_w_cont_normal_ul", tetraPhy.NormalUplinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, True),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_without_cont_prior", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, True, False),
        (sp.ISOLATED_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_without_ramp_bypss", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (True, False, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
    ("con_sync_dl_with_neither", tetraPhy.SyncContDownlinkBurst, 1, tetraPhy.SyncContDownlinkBurst, 3, (False, False, True),
        (sp.START_BURST, sp.MIDDLE_BURST, sp.MIDDLE_BURST, sp.END_BURST)),
]


# Following test is supposed to verify the expansive number of combinations and settings for burstblock continuity, i.e,
# not ramping up/down between contiguous burst blocks, that part of code is 4 functions deep so it requires strict testing
@pytest.mark.parametrize(("name,prior_burst_type,prior_length,target_burst_type,target_length, target_arguments, res_stream_pos"),
                         CONT_PRIOR_CASES,ids=lambda x: x if isinstance(x, str) else None)
def test_continuous_with_prior_behaviour(name, prior_burst_type, prior_length,
                                         target_burst_type, target_length, target_arguments,
                                         res_stream_pos):

    # 1. Schedule the preceeding burst(s)
    tetra_streamer = tetraStream.BurstStreamBuilder()
    prior_burst_channels = get_default_burst_channels(prior_burst_type, prior_length)

    schedule_wrap(tetra_streamer, prior_burst_type, prior_burst_channels,
                  allow_ms_adjacent_slot_ramp_bypass=target_arguments[0],
                  bs_timeshare_adj_slot_ramp_bypass=target_arguments[1],
                  continuous_with_prior_blocks=target_arguments[2])

    # 2. Schedule the target burst(s)
    target_burst_channels = get_default_burst_channels(target_burst_type, target_length)

    schedule_wrap(tetra_streamer, target_burst_type, target_burst_channels,
                  allow_ms_adjacent_slot_ramp_bypass=target_arguments[0],
                  bs_timeshare_adj_slot_ramp_bypass=target_arguments[1],
                  continuous_with_prior_blocks=target_arguments[2])

    burst_list = tetra_streamer.get_scheduled_bursts(return_all_future_bursts=True,
                                                     increment_time_on_call=False)

    assert len(burst_list) == len(res_stream_pos)

    for i, burst in enumerate(burst_list):
        assert burst.stream_position == res_stream_pos[i]
    
    del burst_list
    del tetra_streamer
    del prior_burst_channels
    del target_burst_channels


def test_continuous_time_behaviour():
    # In this test we want to ensure that if if previously scheduled bursts are completely compatible with the currently
    # ones attempting to be scheduled, because of a time gap in scheduling no contiunity occurs between them because 
    # they are not contiguous even if subsequent, also verifys that scheduled schedules at requested times
    tetra_streamer = tetraStream.BurstStreamBuilder()
    prior_burst_type = tetraPhy.NormalUplinkBurst
    prior_burst_channels = get_default_burst_channels(prior_burst_type, 2)

    schedule_wrap(tetra_streamer, prior_burst_type, prior_burst_channels,
                    allow_ms_adjacent_slot_ramp_bypass=True,
                    bs_timeshare_adj_slot_ramp_bypass=True,
                    continuous_with_prior_blocks=True)

    target_burst_type = tetraPhy.NormalUplinkBurst
    target_burst_channels = get_default_burst_channels(target_burst_type, 3)
    sched_time = tetraPhy.TDMATime()
    sched_time += (4*2)
    tetra_streamer.schedule_bursts(target_burst_type,
                                   target_burst_channels,
                                   None, sched_time,
                                   ms_adj_slot_ramp_bypass=True,
                                   bs_timeshare_adj_slot_ramp_bypass=True,
                                   continuous_with_prior_blocks=True,
                                   forced_scheduling=True,
                                   fill_empty_channels=True)
    burst_list = tetra_streamer.get_scheduled_bursts(return_all_future_bursts=True,
                                                     increment_time_on_call=False)

    res_stream_pos = (sp.START_BURST, sp.END_BURST, sp.START_BURST, sp.MIDDLE_BURST, sp.END_BURST)
    res_time_vals = (tetraPhy.TDMATime(frame=1, timeslot=1),
                     tetraPhy.TDMATime(frame=1,timeslot=2),
      # Skipped timeslot as part of time
                     tetraPhy.TDMATime(frame=2, timeslot=1),
                     tetraPhy.TDMATime(frame=2, timeslot=2),
                     tetraPhy.TDMATime(frame=2, timeslot=3))
    assert len(burst_list) == len(res_stream_pos)

    for i, burst in enumerate(burst_list):
        assert burst.stream_position == res_stream_pos[i]
        assert burst.start_time == res_time_vals[i]

def test_collision_detection():
    tetra_streamer = tetraStream.BurstStreamBuilder()
    prior_burst_type = tetraPhy.ControlUplink
    prior_burst_channels = get_default_burst_channels(prior_burst_type, 1)
    prior_sched_time = tetraPhy.TDMATime(frame=2, timeslot=1)
    tetra_streamer.schedule_bursts(prior_burst_type,
                                    prior_burst_channels,
                                    None, prior_sched_time,
                                    ms_adj_slot_ramp_bypass=True,
                                    bs_timeshare_adj_slot_ramp_bypass=True,
                                    continuous_with_prior_blocks=True,
                                    forced_scheduling=True,
                                    fill_empty_channels=True)

    target_burst_type = tetraPhy.NormalUplinkBurst
    target_burst_channels = get_default_burst_channels(target_burst_type, 5)
    sched_time = tetraPhy.TDMATime(frame=1, timeslot=1)

    with pytest.raises(tetraStream.ScheduledBurstCollisionError):
        tetra_streamer.schedule_bursts(target_burst_type,
                                        target_burst_channels,
                                        None, sched_time,
                                        ms_adj_slot_ramp_bypass=True,
                                        bs_timeshare_adj_slot_ramp_bypass=True,
                                        continuous_with_prior_blocks=True,
                                        forced_scheduling=True,
                                        fill_empty_channels=True)
        