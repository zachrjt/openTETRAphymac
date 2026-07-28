
"""
tetra_stream.py contains the classes and related functions that are used to implement simulation
of TETRA burst streaming between a BS/MS -> MS/BS in a simulation context, it encapsulates the primative
physical_channel Burst types and LogicalChannel types allowing for a single object to be used in each step
as part of block streaming.

Allows for comparision of BER, MER, EVM at the end of TX->CH->RCVR chain without external data bookkeeping
"""
###################################################################################################
import textwrap
from typing import List
from collections import deque

from numpy.typing import NDArray
from numpy import uint8, complex128, complex64

from .physical_channels import Burst, PhysicalChannel, TDMATime
from .logical_channels import LogicalChannelVD
from .constants import StreamPosition, PhyType, OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY, \
                       OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY

TETRA_DEFAULT_TP_PHY = PhysicalChannel(PhyType.TRAFFIC_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

TETRA_DEFAULT_CP_PHY = PhysicalChannel(PhyType.CONTROL_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

TETRA_DEFAULT_UP_PHY = PhysicalChannel(PhyType.UNASGN_CHANNEL, 1, False, OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY,
                                       OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY)

###################################################################################################


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


class BurstStreamBuilder():
    __slots__ = (
        "mode",
        "_queue",

        "current_tetra_time",
        "max_scheduled_tetra_time"
    )
    mode: bool
    _queue: deque[BurstBlock]

    current_tetra_time: TDMATime
    max_scheduled_tetra_time: TDMATime

    def __init__(self, rf_channels: tuple[PhysicalChannel, ...] | None,
                 forced: bool = False, tetra_time: TDMATime | None = None):
        self.mode = forced

        if tetra_time is None:
            self.current_tetra_time = TDMATime()
        else:
            self.current_tetra_time = tetra_time.copy()

        self._queue = deque()

    def construct_burst_block(self, burst_type: type[Burst],
                              input: tuple[list[LogicalChannelVD] | None, ...],
                              phy_channel: PhysicalChannel | None,
                              allow_ms_adjacent_slot_ramp_bypass: bool = False,
                              continuous_with_prior_blocks: bool = True,
                              advance_slot_counter: bool = True,
                              replicate_and_fill: bool = False) -> List[BurstBlock] | None:

        input_local = [None if x is None else list(x) for x in input]
        burst_blocks: list[BurstBlock] | None = None
        input_blocks_total_length = [0] * len(input)

        for i, input_blocks in enumerate(input_local):
            if input_blocks is not None:
                for ch in input_blocks:
                    # Verify that type five bits are present within logical channel
                    if not ch.has_type_5_blocks:
                        # Since there is no bits ready for burst block insertion, generate the data
                        if replicate_and_fill:
                            ch.encode_type5_bits(ch.generate_rnd_input(1))
                        else:
                            raise ValueError(f"Passed logical channel to BurstScheduler: {ch}, has no type 5 bits")
                    input_blocks_total_length[i] += ch.type_5_blocks.shape[0]

        max_number_of_bursts = max(input_blocks_total_length)
        if min(input_blocks_total_length) == 0:
            return None

        burst_blocks = []
        if replicate_and_fill:
            for i, input_blocks in enumerate(input_local):
                if input_blocks is not None:
                    num_blocks_to_add = max_number_of_bursts - input_blocks_total_length[i]
                    for _ in range(num_blocks_to_add):
                        # Since we are replicate_and_fill mode, we want to replicate the channel types to ensure
                        # we fill the maximum number of burst given the original input logical channel blocks
                        # i.e. if we pass traffic block with m=5, but only two other control blocks, we
                        # generate extra control blocks and fill with dummy data
                        seed = input_blocks[0].seed_seq.spawn(1)[0]
                        input_local[i].append(type(input_blocks[0])(seed_seq=seed))  # type: ignore
                        bits = input_local[i][-1].generate_rnd_input(1)  # type: ignore
                        input_local[i][-1].encode_type5_bits(bits)  # type: ignore

        # Handle the rf_channel argument, can accept 3 types, 2 of which require some lookup
        if isinstance(phy_channel, PhysicalChannel):
            pass
        elif phy_channel is None:
            # passed None, in this case find the oldest (lowest index) stored channel that matches the first ALLOWED_PHY
            # for the given burst
            phy_type_for_burst = next(iter(burst_type.ALLOWED_PHY))
            if phy_type_for_burst == PhyType.CONTROL_CHANNEL:
                phy_channel = TETRA_DEFAULT_CP_PHY
            elif phy_type_for_burst == PhyType.TRAFFIC_CHANNEL:
                phy_channel = TETRA_DEFAULT_TP_PHY
            else:
                phy_channel = TETRA_DEFAULT_UP_PHY
        else:
            raise TypeError(f"BurstScheduler passed rf_channel must be None or PhysicalChannel, got "
                            f"{type(phy_channel).__name__}")

        for i in range(min(input_blocks_total_length)):
            burst_container = burst_type(phy_channel=phy_channel,
                                         tetra_time=self.max_scheduled_tetra_time,
                                         forced=replicate_and_fill)
        return None
