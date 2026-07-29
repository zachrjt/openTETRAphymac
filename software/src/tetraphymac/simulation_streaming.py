
"""
tetra_stream.py contains the classes and related functions that are used to implement simulation
of TETRA burst streaming between a BS/MS -> MS/BS in a simulation context, it encapsulates the primative
physical_channel Burst types and LogicalChannel types allowing for a single object to be used in each step
as part of block streaming.

Allows for comparision of BER, MER, EVM at the end of TX->CH->RCVR chain without external data bookkeeping
"""
###################################################################################################
import textwrap
from bisect import bisect_left
from dataclasses import dataclass, field

from numpy.typing import NDArray
from numpy import uint8, complex128, complex64

from .physical_channels import Burst, PhysicalChannel, TDMATime, RFCarrier, NormalUplinkBurst, \
                               NormalContDownlinkBurst, SyncContDownlinkBurst
from .logical_channels import LogicalChannelVD
from .constants import StreamPosition, PhyType, OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY, \
                       OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY, TIMESLOT_SUBSLOT_LENGTH, BurstContinuity, \
                       TETRA_RAMP_BOOLS_FROM_STREAM_POSITION, TETRA_STREAM_POSITION_FROM_RAMP_BOOLS

TETRA_DEFAULT_TP_PHY = PhysicalChannel(PhyType.TRAFFIC_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

TETRA_DEFAULT_CP_PHY = PhysicalChannel(PhyType.CONTROL_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

TETRA_DEFAULT_UP_PHY = PhysicalChannel(PhyType.UNASGN_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

TETRA_DEFAULT_PHY_DICT = {PhyType.CONTROL_CHANNEL: TETRA_DEFAULT_CP_PHY,
                          PhyType.TRAFFIC_CHANNEL: TETRA_DEFAULT_TP_PHY,
                          PhyType.UNASGN_CHANNEL: TETRA_DEFAULT_UP_PHY}

TETRA_BURST_TYPES_THAT_SUPPORT_CONTINUITY = (NormalUplinkBurst, NormalContDownlinkBurst, SyncContDownlinkBurst)
###################################################################################################


class ScheduledBurstCollisionError(Exception):
    """Raised when an attempt to schedule a burst happens that collides with already scheduled bursts"""
    pass


def _validate_stream_position(ramp_indices: tuple[int, int], stream_position: StreamPosition) -> bool:
    validation_state = False

    if stream_position == StreamPosition.ISOLATED_BURST:
        if ramp_indices[0] > 0 and ramp_indices[1] > 0:
            validation_state = True
    elif stream_position == StreamPosition.START_BURST:
        if ramp_indices[0] > 0 and ramp_indices[1] == 0:
            validation_state = True
    elif stream_position == StreamPosition.MIDDLE_BURST:
        if ramp_indices[0] == 0 and ramp_indices[1] == 0:
            validation_state = True
    elif stream_position == StreamPosition.END_BURST:
        if ramp_indices[0] == 0 and ramp_indices[1] > 0:
            validation_state = True

    return validation_state


class BurstBlock():
    __slots__ = (
        "burst",
        "logical_channels",
        "logical_ch_block_indices",

        "subslot",
        "stream_position",
        "forced",

        "modulation_bits",
        "modulation_symbols",
        "samples"
    )
    burst: Burst
    logical_channels: tuple[LogicalChannelVD, ...]
    logical_ch_block_indices: tuple[int, ...]

    subslot: bool
    stream_position: StreamPosition
    forced: bool

    modulation_bits: NDArray[uint8]
    modulation_symbols: NDArray[complex64] | None
    samples: NDArray[complex128] | None

    def __init__(self, burst: Burst, logical_channels: tuple[LogicalChannelVD, ...],
                 burst_indices: tuple[int, ...], stream_position: StreamPosition,
                 modulation_bits: NDArray[uint8]):
        self.burst = burst
        self.forced = burst.forced
        self.logical_channels = logical_channels
        if len(self.logical_channels) != len(burst_indices):
            raise ValueError(f"Passed tuple of size: {len(self.logical_channels)},"
                             f" but given burst_indices of size: {len(burst_indices)}, expected same.")
        self.logical_ch_block_indices = burst_indices

        if not _validate_stream_position((self.burst.start_ramp_period, self.burst.end_ramp_period), stream_position):
            raise ValueError(f"Passed stream position of: {stream_position}, but ramping indices were:"
                             f"{(self.burst.start_ramp_period, self.burst.end_ramp_period)}, unexpected")

        self.stream_position = stream_position
        self.subslot = True if self.burst.subslot_width == 1 else False

        self.modulation_bits = modulation_bits

        self.modulation_symbols = None
        self.samples = None

    @property
    def has_samples(self):
        return self.samples is not None

    @property
    def has_symbols(self):
        return self.modulation_symbols is not None

    def __repr__(self) -> str:
        r_str = f"BurstBlock, burst type:{self.burst.__class__.__name__}, subslot?={self.subslot},"
        r_str += f" link direction={self.burst.link_direction}, RF channel type={self.burst.phy_channel},"
        r_str += f" stream position={self.stream_position}, logical channels={self.logical_channels}"
        r_str = textwrap.fill(r_str, width=70)
        return r_str

###################################################################################################


@dataclass(order=True, slots=True)
class ScheduledBurstBlock:
    time: TDMATime
    duration: int = field(compare=False)    # Duration in number of subslots

    carrier: RFCarrier = field(compare=False)
    burst_block: BurstBlock = field(compare=False)

    @property
    def occupied_times(self) -> tuple[TDMATime, ...]:
        return tuple(self.time + i for i in range(self.duration))

    @property
    def end_time(self) -> TDMATime:
        return self.time + self.duration

###################################################################################################


class BurstStreamBuilder():
    __slots__ = (
        "mode",
        "_queue",

        "current_tetra_time",
    )
    mode: bool
    _queue: list[ScheduledBurstBlock]

    current_tetra_time: TDMATime

    def next_available_time_for_carrier(self, carrier: RFCarrier, subslot_duration: int) -> TDMATime:
        if subslot_duration not in (1, 2):
            raise ValueError(f"Expected subslot duration of 1 or 2, got: {subslot_duration}")

        for block in reversed(self._queue):
            if block.carrier.id == carrier.id:
                next_time = block.end_time
                if subslot_duration == TIMESLOT_SUBSLOT_LENGTH and next_time.subslot != 1:
                    next_time += 1
                return next_time

        next_time = self.current_tetra_time
        if subslot_duration == TIMESLOT_SUBSLOT_LENGTH and next_time.subslot != 1:
            next_time += 1

        return next_time

    @staticmethod
    def revise_burst_ramping_to_continuous(target_block: BurstBlock) -> bool:
        # 1. Determine and revise ramping bool tuple to be continuous with subsequent burst block
        #    and determine revised stream position
        old_ramp_bools = TETRA_RAMP_BOOLS_FROM_STREAM_POSITION[target_block.stream_position]
        new_ramp_bools = (old_ramp_bools[0], False)
        new_stream_position = TETRA_STREAM_POSITION_FROM_RAMP_BOOLS[new_ramp_bools]

        # 2. Regenerate the modulation bits
        if isinstance(target_block.burst, TETRA_BURST_TYPES_THAT_SUPPORT_CONTINUITY):
            logical_ch_views: list[LogicalChannelVD] = []
            for i, ch in enumerate(target_block.logical_channels):
                logical_ch_views.append(ch.get_type_5_block_view(target_block.logical_ch_block_indices[i]))
            # I tried a few ways to get mypy to play ball but unless i write out explict cases for every length
            # of logical_ch_views, it freaks out and cant handle a dynamic length unpacking even if i check the
            # subsequent length dynamically
            new_mod_bits = target_block.burst.construct_burst_sequence(*logical_ch_views,  # type: ignore[arg-type]
                                                                       ramp_up_down_state=(
                                                                        new_ramp_bools))  # type: ignore[misc]
            target_block.modulation_bits = new_mod_bits
            target_block.stream_position = new_stream_position
            return True
        else:
            return False

    def check_scheduled_bursts_for_collisions(self, carrier: RFCarrier,
                                              start_time: TDMATime,
                                              end_time: TDMATime) -> bool:
        upper = bisect_left(self._queue, end_time, key=lambda x: x.time)  # find index of first block.time < end_time
        lower = bisect_left(self._queue, start_time - (TIMESLOT_SUBSLOT_LENGTH - 1), key=lambda x: x.time)
        # Durations are either 1 or 2 (`TIMESLOT_SUBSLOT_LENGTH`)
        # Therefore any burst that could overlap must start no earlier than (start_time - 1)
        for block in reversed(self._queue[lower:upper]):
            if block.carrier.id == carrier.id:
                return True
        return False

    def handle_prior_contiguous_burst_continuity(self, carrier: RFCarrier, start_time: TDMATime,
                                                 burst_type: type[Burst],
                                                 allow_ms_adjacent_slot_ramp_bypass: bool) -> bool:

        # 1. Check if we allow for continuous bursts with our current burst_type before anything else
        if burst_type.CONTINUITY_MODE == BurstContinuity.ISOLATED or (
           burst_type.CONTINUITY_MODE == BurstContinuity.OPTIONAL and not allow_ms_adjacent_slot_ramp_bypass):
            return False
        # 2. Determine if there is a preceeding burst that is adjacent in end_time to our burst and has same carrier
        lower = bisect_left(self._queue, (start_time - TIMESLOT_SUBSLOT_LENGTH), key=lambda x: x.time)
        upper = bisect_left(self._queue, start_time, key=lambda x: x.time)
        preceding_block = None
        for block in self._queue[lower:upper]:
            # block(s) are within the preceeding timeslot
            if block.end_time == start_time and block.carrier.id == carrier.id:
                # block ends is contiguous and shares rf carrier
                preceding_block = block
                break

        if preceding_block is None:
            return False
        # Check if the adjacent burst is compatible and or allows for continuous bursts
        compatible = (burst_type.CONTINUITY_BURST_TYPE in
                      preceding_block.burst_block.burst.CONTINUITY_COMPATIBLE_BURST_TYPES)

        # 3. If compatible, revise the preceeding contiguous burst to be continuous
        match preceding_block.burst_block.burst.CONTINUITY_MODE:
            case BurstContinuity.REQUIRED:
                if compatible:
                    return self.revise_burst_ramping_to_continuous(preceding_block.burst_block)
            case BurstContinuity.OPTIONAL:
                if allow_ms_adjacent_slot_ramp_bypass:
                    if compatible:
                        return self.revise_burst_ramping_to_continuous(preceding_block.burst_block)
            case BurstContinuity.ISOLATED:
                pass

        return False

    def __init__(self, rf_channels: tuple[PhysicalChannel, ...] | None,
                 forced: bool = False, tetra_time: TDMATime | None = None):
        self.mode = forced

        if tetra_time is None:
            self.current_tetra_time = TDMATime()
        else:
            self.current_tetra_time = tetra_time.copy()

        self._queue = []

    def construct_burst_blocks(self, burst_type: type[Burst],
                               input_logical_ch: tuple[list[LogicalChannelVD | None], ...],
                               phy_channel: PhysicalChannel | None,
                               start_time: TDMATime | None, *,
                               allow_ms_adjacent_slot_ramp_bypass: bool = False,
                               continuous_with_prior_blocks: bool = True,
                               forced: bool = False,
                               fill_empty_channels: bool = False) -> list[BurstBlock] | None:

        # 1. Handle processing input logical channels and blocks to determine total length
        local_logical_ch_input: tuple[list[LogicalChannelVD | None], ...] = tuple(list(x) for x in input_logical_ch)
        stream_lengths = [0] * len(input_logical_ch)

        # Remove "trailing" None's, only preceeding Nones are useful in determining output length
        for stream in local_logical_ch_input:
            while stream and stream[-1] is None:
                stream.pop()

        for i, input_blocks in enumerate(local_logical_ch_input):
            for ch_list in input_blocks:
                if ch_list is None:
                    stream_lengths[i] += 1
                if ch_list is not None:
                    # Verify that type five bits are present within logical channel
                    if not ch_list.has_type_5_blocks:
                        # Since there is no bits ready for burst block insertion, generate the data
                        if fill_empty_channels:
                            ch_list.encode_type5_bits(ch_list.generate_rnd_input(1))
                        else:
                            raise ValueError(f"Passed logical channel to BurstScheduler: {ch_list}, has no type 5 bits")
                    stream_lengths[i] += ch_list.type_5_blocks.shape[0]

        if not stream_lengths:
            return None
        output_length = max(stream_lengths)

        if output_length == 0:
            return None

        # 2. Handle physical channel and timing input parameters
        if phy_channel is None:
            if not forced:
                raise ValueError("BurstStreamBuilder was passed None for phy_channel, but forced=False, invalid"
                                 " input argument combination, expected specified phy_channel or forced=True")
            phy_channel = TETRA_DEFAULT_PHY_DICT[burst_type.DEFAULT_PHY]

        if start_time is None:
            start_time = self.next_available_time_for_carrier(phy_channel.carrier, burst_type.subslot_width)
        else:
            end_time = start_time + ((output_length - 1) * TIMESLOT_SUBSLOT_LENGTH) + burst_type.subslot_width
            collision = self.check_scheduled_bursts_for_collisions(phy_channel.carrier, start_time, end_time)
            if collision:
                raise ScheduledBurstCollisionError("Cannot schedule burst(s) on carrier: "
                                                   f"{phy_channel.carrier}"
                                                   f", starting at: {start_time} and ending at: {end_time}")

        # 3. Sort through logical channels to allow for easier use in burst creation argument unpacking
        logical_chs_list: list[list[LogicalChannelVD]] = [[] for _ in range(output_length)]
        logical_ch_views_list: list[list[LogicalChannelVD]] = [[] for _ in range(output_length)]
        logical_ch_block_indices_list: list[list[int]] = [[] for _ in range(output_length)]

        for i in range(len(local_logical_ch_input)):
            burst_index = 0
            ch_list_index = 0
            list_len = len(local_logical_ch_input[i])

            while burst_index < output_length and ch_list_index < list_len:
                ch = local_logical_ch_input[i][ch_list_index]
                if ch is not None:
                    for block_index in range(ch.type_5_blocks.shape[0]):
                        logical_chs_list[burst_index].append(ch)
                        logical_ch_views_list[burst_index].append(ch.get_type_5_block_view(block_index))
                        logical_ch_block_indices_list[burst_index].append(block_index)
                        burst_index += 1
                    ch_list_index += 1

                else:
                    ch_list_index += 1
                    burst_index += 1

        logical_chs: list[tuple[LogicalChannelVD, ...]] = [tuple(x) for x in logical_chs_list]
        logical_ch_views: list[tuple[LogicalChannelVD, ...]] = [tuple(x) for x in logical_ch_views_list]
        logical_ch_block_indices: list[tuple[int, ...]] = [tuple(x) for x in logical_ch_block_indices_list]

        # 4. Determine if start burst is continuous with previous one
        if continuous_with_prior_blocks:
            block_1_cont_with_prior = self.handle_prior_contiguous_burst_continuity(phy_channel.carrier,
                                                                                    start_time,
                                                                                    burst_type,
                                                                                    allow_ms_adjacent_slot_ramp_bypass)
        else:
            block_1_cont_with_prior = False

        if burst_type.CONTINUITY_MODE == BurstContinuity.OPTIONAL and allow_ms_adjacent_slot_ramp_bypass:
            burst_blocks_continuous = True
        elif burst_type.CONTINUITY_MODE == BurstContinuity.REQUIRED:
            burst_blocks_continuous = True
        else:
            burst_blocks_continuous = False

        burst_blocks: list[BurstBlock] = []

        for i in range(output_length):
            # 1. Create burst
            sched_time = start_time + (i * burst_type.subslot_width)
            burst = burst_type(phy_channel=phy_channel, tetra_time=sched_time, forced=forced)
            # 2. Determine the relevant stream position descriptor
            block_stream_pos = None
            if i == 0:
                if burst_blocks_continuous:
                    if block_1_cont_with_prior:
                        if output_length == 1:
                            block_stream_pos = StreamPosition.END_BURST
                        else:
                            block_stream_pos = StreamPosition.MIDDLE_BURST
                    else:
                        if output_length == 1:
                            block_stream_pos = StreamPosition.ISOLATED_BURST
                        else:
                            block_stream_pos = StreamPosition.START_BURST
                else:
                    block_stream_pos = StreamPosition.ISOLATED_BURST
            elif i == output_length-1:
                if burst_blocks_continuous:
                    block_stream_pos = StreamPosition.END_BURST
                else:
                    block_stream_pos = StreamPosition.ISOLATED_BURST
            else:
                if burst_blocks_continuous:
                    block_stream_pos = StreamPosition.MIDDLE_BURST
                else:
                    block_stream_pos = StreamPosition.ISOLATED_BURST
            ramp_bools = TETRA_RAMP_BOOLS_FROM_STREAM_POSITION[block_stream_pos]

            # 3. Generate Burst modulation bits
            mod_bits = burst.construct_burst_sequence(*logical_ch_views[i],  # type: ignore[arg-type, attr-defined]
                                                      ramp_up_down_state=ramp_bools)  # type: ignore[misc]
            burst_block = BurstBlock(burst, logical_chs[i], logical_ch_block_indices[i],
                                     block_stream_pos, mod_bits)  # type: ignore[arg-type]
            burst_blocks.append(burst_block)

        return burst_blocks

###################################################################################################
