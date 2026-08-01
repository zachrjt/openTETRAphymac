
"""
tetra_stream.py contains the classes and related functions that are used to implement simulation
of TETRA burst streaming between a BS/MS -> MS/BS in a simulation context, it encapsulates the primative
physical_channel Burst types and LogicalChannel types allowing for a single object to be used in each step
as part of block streaming.

Allows for comparision of BER, MER, EVM at the end of TX->CH->RX chain without external data bookkeeping
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
from .constants import StreamPosition, PhysicalChannelType, OPENTETRAPHYMAC_DEFAULT_DL_FREQUENCY, \
                       OPENTETRAPHYMAC_DEFAULT_UL_FREQUENCY, TIMESLOT_SUBSLOT_LENGTH, BurstContinuity, \
                       TETRA_RAMP_BOOLS_FROM_STREAM_POSITION, TETRA_STREAM_POSITION_FROM_RAMP_BOOLS

TETRA_DEFAULT_TP_PHY = PhysicalChannel(PhysicalChannelType.TRAFFIC_CHANNEL, 1, False,
                                       OPENTETRAPHYMAC_DEFAULT_UL_FREQUENCY, OPENTETRAPHYMAC_DEFAULT_DL_FREQUENCY)

TETRA_DEFAULT_CP_PHY = PhysicalChannel(PhysicalChannelType.CONTROL_CHANNEL, 1, False,
                                       OPENTETRAPHYMAC_DEFAULT_UL_FREQUENCY, OPENTETRAPHYMAC_DEFAULT_DL_FREQUENCY)

TETRA_DEFAULT_UP_PHY = PhysicalChannel(PhysicalChannelType.UNASGN_CHANNEL, 1, False,
                                       OPENTETRAPHYMAC_DEFAULT_UL_FREQUENCY, OPENTETRAPHYMAC_DEFAULT_DL_FREQUENCY)

TETRA_DEFAULT_PHY_DICT = {PhysicalChannelType.CONTROL_CHANNEL: TETRA_DEFAULT_CP_PHY,
                          PhysicalChannelType.TRAFFIC_CHANNEL: TETRA_DEFAULT_TP_PHY,
                          PhysicalChannelType.UNASGN_CHANNEL: TETRA_DEFAULT_UP_PHY}

TETRA_BURST_TYPES_THAT_SUPPORT_CONTINUITY = (NormalUplinkBurst, NormalContDownlinkBurst, SyncContDownlinkBurst)
###################################################################################################


class ScheduledBurstCollisionError(Exception):
    """Raised when an attempt to schedule a burst happens that collides with already scheduled burst(s)"""


def _validate_stream_position(ramp_indices: tuple[int, int], stream_position: StreamPosition) -> bool:
    """
    Validates the passed `stream_position` argument with respect to the passed `ramp_indices` tuple pair,
    ensuring that they match, which is essiental to prevent FIR transients due to a lack of ramping up/down

    :param ramp_indices: Ramping indices specify how many symbol periods from SN0, and SNmax are used for ramping to 0
    :type ramp_indices: Tuple[int, int]
    :param stream_position: stream position descriptor, either ISOLATED, START, MIDDLE, or END
    :type stream_position: StreamPosition enum type
    :return: True if `stream_position` matches the `ramp_indices` tuple pair, False otherwise
    :rtype: bool
    """
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


@dataclass(order=True, slots=True, init=False)
class BurstBlock():
    """
    `BurstBlock` is **the fundamental** openTETRAphymac simulation layer dataclass that encapsulates regular `Burst`
    objects alongside their resulting bit, symbol, tx/rx data, and contextual information.

    When high-level openTETRAphymac classes pass burst data to one another in the
    [data -> TX -> CHANNEL -> RX -> data] system chain, they use `BurstBlock`'s to add or modify data contained within.

    Compared to handling data discretely, `BurstBlock`'s offer easier usage via the following means:
        1. Ordered dataclass via `start_time` and RFCarrier `channel_number` for ties, allowing for easier scheduling,
         sorting, and handling when dealing with multiple MS's or BS's
        2. They contain the original input data; the crc'ed, encoded, interleaved, logical channel data;
         burst modulation bit sequence; modulation symbol sequence; and the transmitted/receiver data. This means
         it is very easier to handle BER, MER, and other comparisions as the data moves through the transmitter-reciever
         chain

    **Usage**:
        - A `BurstBlock` can be specified manually or can be generated automatically using a `BurstStreamBuilder`
         instance, by passing the type of burst desired, the input logical channels, and the start time.
            - In this stage of life the `logical_channels`, `modulation_bits` and `burst` type fields are filled

        - A populated `BurstBlock` or a list of them, either generated manually or gotten from a
         `BurstStreamBuilder` instance queue via `BurstStreamBuilder.get_scheduled_bursts(...)` are then passed to a
         standalone `transmitter` instance or MS or BS instance to transmit.
            - In this stage, the resulting `transmitter` will generate and add `modulation_symbols` and tx `samples`

        - A transmitted `BurstBlock` intend to be receiver by a target receiver can be affected by a channel by
         passing it to a `ChannelSimulator` instance built for that transmiter->receiver RF link.
            - In this stage, the `samples` get applied propgation losses, doppler spread, delay and delay co-channel
              ISI, and adjacent channel interference from CW's or TETRA signals.

        - A channel affected, transmitted, `BurstBlock` is then received by a `Receiver` instance, either standalone
         or within a BS/MS. The receiver attempts to receive and demodulate the signal and is able to compare the
         demodulated bits and messages against the content within the `BurstBlock`
    """
    burst: Burst = field(compare=True)
    logical_channels: tuple[LogicalChannelVD, ...] = field(compare=False)
    logical_ch_block_indices: tuple[int, ...] = field(compare=False)

    subslot: bool = field(compare=False)
    stream_position: StreamPosition = field(compare=False)
    forced: bool = field(compare=False)

    modulation_bits: NDArray[uint8] = field(compare=False)
    modulation_symbols: NDArray[complex64] | None = field(compare=False)
    samples: NDArray[complex128] | None = field(compare=False)

    def __init__(self, burst: Burst, logical_channels: tuple[LogicalChannelVD, ...],
                 burst_indices: tuple[int, ...], stream_position: StreamPosition,
                 modulation_bits: NDArray[uint8]):
        """
        Initialzes a BurstBlock data object using the input arguments

        :param burst: Burst instance
        :type burst: Burst
        :param logical_channels: A tuple of the required logical channels that were used in the creation of the burst
        :type logical_channels: Tuple[LogicalChannelVD, ...]
        :param logical_ch_block_indices: A tuple of int indices values for the used block from the logical_channels
         type_5_block output data, which is a two array, indices can be zero, should be one value for every
         logical_channel passed
        :type logical_ch_block_indices: Tuple[int, ...]
        :param stream_position: stream position descriptor, either ISOLATED, START, MIDDLE, or END
        :type stream_position: StreamPosition enum type
        :param modulation_bits: The output modulation bits from `burst.construct_burst_sequence(..)`
        :type modulation_bits: NDArray[uint8]
        """
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
        self.subslot = self.burst.subslot_width == 1

        self.modulation_bits = modulation_bits

        self.modulation_symbols = None
        self.samples = None

    @property
    def has_samples(self):
        """
        True if `samples` of this BurstBlock instance is populated, i.e. has been transmitted, False otherwise
        """
        return self.samples is not None

    @property
    def has_symbols(self):
        """
        True if `symbols` of this BurstBlock instance is populated, i.e. has been transmitted, False otherwise
        """
        return self.modulation_symbols is not None

    @property
    def duration(self) -> int:
        """
        The duration of the burst in number of subslots, an int
        """
        return self.burst.subslot_width

    @property
    def occupied_times(self) -> tuple[TDMATime, ...]:
        """
        Returns a tuple of `TDMATime`'s for every subslot that the burst occupies
        """
        return tuple(self.burst.start_time + i for i in range(self.duration))

    @property
    def end_time(self) -> TDMATime:
        """
        The end time of the burst, a `TDMATime` object return
        """
        return self.burst.end_time

    @property
    def start_time(self) -> TDMATime:
        """
        The start time of the burst, a `TDMATime` object return
        """
        return self.burst.start_time

    @property
    def carrier(self) -> RFCarrier:
        """
        The `RFCarrier` of the burst, a `RFCarrier` object return
        """
        return self.burst.rf_carrier

    def __repr__(self) -> str:
        r_str = f"BurstBlock, burst type:{self.burst.__class__.__name__},"
        r_str += f" link direction={self.burst.link_direction.value}, channel type={self.burst.phy_channel_type.value},"
        r_str += f" RF carrier number: {self.burst.rf_carrier.channel_number},"
        r_str += f" stream position={self.stream_position.value}, logical channels="
        r_str += f"({self.logical_channels[0]}"
        for i in range(1, len(self.logical_channels)):
            r_str += f",{self.logical_channels[i]}"
        r_str += "),"
        r_str += f" Start Time: {self.burst.start_time} -- End Time: {self.burst.end_time}"
        r_str = textwrap.fill(r_str, width=100)
        return r_str

###################################################################################################


class BurstStreamBuilder():
    """
    `BurstStreamBuilder` is a openTETRAphymac simulation layer coordinator class that handles generating BurstBlock's,
    scheduling them in a TETRA protocal naive way, and returning scheduled bursts when a caller asks from them at the
    correct time.

    **Note 1:** `BurstStreamBuilder` is not a replacement for a lower MAC scheduler, it does not perform TETRA protocal
    "aware" scheduling meaning it does care about the configured times of bursts, the physical channel compatibility,
    or other MAC layer concerns.
    - Thus `BurstStreamBuilder` can function when scheduling in two modes: `forced_scheduling = True` or `False`,
     in True mode the builder tells generated bursts to not care about violations of timing or PhysicalChannel timeslot
     allocation, this mode is expected to be used in standalone / user config mode, when the caller just wants bursts.
    - In `forced_scheduling = False`, bursts will perform checks and thus it is expected that the caller is a lower MAC
     layer implementation that will insure that bursts are scheduled correctly and thus `BurstStreamBuilder` plays the
     role of a simple constructor and queue, with `PhysicalChannel`'s and start/end times being specified by the MAC.



    `BurstStreamBuilder` contains a handful of methods used to assist in it's goal, the primary external methods to be
    called are:
    1. `schedule_bursts`: A method that when passed data regarding the PhysicalChannel, Start Time, Burst Type, Logical
     Channels, schedules burst(s) into `BurstStreamBuilder`'s internal `queue`.  It handles detecting collisions and is
     can be configured with arguments to automatically use default RFchannels, force scheduling despite timing rule
     violation of bursts, automatically fill empty passed logical channels, detect, modify, and continue burst
     continuity between already scheduled compatible bursts and contigous bursts attempting to be scheduled.
    2. `get_scheduled_bursts`: A method used to "pop" scheduled bursts out of `BurstStreamBuilder`, capable of returning
     burst(s) scheduled at a specified time or current time to a specified end time or number of timeslots into the
     future beyond the start time
    """
    __slots__ = (
        "_queue",

        "current_tetra_time",
    )
    _queue: list[BurstBlock]        # Ordered list of BurstBlocks that are scheduled
    current_tetra_time: TDMATime    # The current time of the BurstStreamBuilder

    def __init__(self, current_time: TDMATime | None = None):
        """
        Initialzes a BurstBlock data object which is set to be at the passed time.

        :param current_time: The current TDMA time for this scheduler, if default/None, generates TDMATime at t=0
        :type current_time: TDMATime | None, default None generates a TDMATime internally that starts at t=0
        """

        if current_time is None:
            self.current_tetra_time = TDMATime()
        else:
            self.current_tetra_time = current_time.copy()

        self._queue = []

    def _next_available_time_for_carrier(self, carrier: RFCarrier, subslot_duration: int) -> TDMATime:
        """
        Determines and returns the next subsequent start time for a burst to be scheduled based on the `carrier`
        specified and the `subslot_duration` of the intended burst type.

        **NOTE**: This function is TETRA protocol naive, it does not verify the timing requirements of the physical
        channel, burst type, nor does it attempt to fit/schedule bursts before already scheduled bursts in the queue.
        It is expected that if the caller wants explicit or optimized scheduling, that it specifies the scheduling times
        for bursts with awareness of timing requirements. The use of this basic scheduler is for intended for basic
        burst data generation and simulation work.

        :param carrier: RFCarrier instance that the caller wants to find the next subsequent time available for
         scheduling a burst, a burst type with duration `subslot_duration`
        :type carrier: RFCarrier
        :param subslot_duration: The duration of the burst type desired to be scheduled in int number of subslots
        :type subslot_duration: int
        :return: the next time available for scheduling based on the given RFCarrier, note that this time is always
         after any previously scheduled bursts even if there is a available gap before that scheduled burst,
         in the case of needing explict scheduling like that caller should specify `start_time` in `schedule_bursts`
        :rtype: TDMATime
        """
        if subslot_duration not in tuple(x for x in range(1, TIMESLOT_SUBSLOT_LENGTH+1)):
            raise ValueError(f"Expected subslot duration in range [1, {TIMESLOT_SUBSLOT_LENGTH}],"
                             f" got: {subslot_duration}")

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
    def _revise_burst_ramping_to_continuous(target_block: BurstBlock) -> bool:
        """
        Given a `target_block` BurstBlock that is scheduled in `_queue`, this function overwrites the modulation bit
        data of the block with new data using the BurstBlock burst and logical channel data but this time configuring
        the ramping to not ramp down at the end of the burst such that the burst will be continuous with a subsequent
        burst attempting to be scheduled.

        **NOTE**: If that the passed `target_block` burst type does not support continuous signal / no ramping, the
        function returns False otherwise it returns True if sucessfull.

        :param target_block: A BurstBlock, passed by reference, that is scheduled in `_queue` which the caller desires
         to have it's modulation bits revised to be continuous with a subsequent burst that BurstStreamBuilder is
         trying to schedule
        :type target_block: BurstBlock
        :return: Returns True if the preceding contiguous block was sucessfully revised to be continuous with a
         subsequent block, otherwise returns False if that block capable of being continuous
        :rtype: bool
        """
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

        return False

    def _check_scheduled_bursts_for_collisions(self, carrier: RFCarrier,
                                               start_time: TDMATime,
                                               end_time: TDMATime) -> bool:
        """
        Given the provided `carrier`, `start_time`, and `end_time`, where `end_time` is inclusive of the duration of the
        [-1]'th burst planned to scheduled, this function checks the already scheduled bursts in `_queue` and determines
        if there is a collision, which is defined as when two bursts are scheduled at the same time **AND** occupy the
        same RFCarrier

        **NOTE**: This function is TETRA protocol naive, it does not verify the timing requirements of the physical
        channel, burst type, it simply checks for if scheduled bursts will overlap in transmission time and RFCarrier
        usage.

        :param carrier: RFCarrier instance that the planned scheduled bursts will use, used to compare against scheduled
         bursts in the `_queue` to determine if a collision (RFCarrier and Time overlap) occurs
        :type carrier: RFCarrier
        :param start_time: The start_time for the burst(s)
        :type start_time: TDMATime
        :param end_time: The end_time for the burst, should include the [-1] burst's duration such that the
         planned burst(s) only exist within [start_time -> end_time), note that an already
         scheduled burst can start on end_time without collision but cannot start before end_time
        :type end_time: TDMATime
        :return: Returns True if a burst already scheduled in `_queue` exists occupies space in between `start_time`
         and `end_time` and has a matching `carrier` value, thus creating a collision
        :rtype: bool
        """
        upper = bisect_left(self._queue, end_time, key=lambda x: x.start_time)  # index of first block.time < end_time
        lower = bisect_left(self._queue, start_time - (TIMESLOT_SUBSLOT_LENGTH - 1), key=lambda x: x.start_time)
        # Durations are either 1 or `TIMESLOT_SUBSLOT_LENGTH` (2 for TETRA)
        # Therefore any burst that could overlap must start no earlier than (start_time - 1)
        for block in reversed(self._queue[lower:upper]):
            if block.carrier.id == carrier.id:
                return True
        return False

    def _handle_prior_contiguous_burst_continuity(self, carrier: RFCarrier, start_time: TDMATime,
                                                  burst_type: type[Burst],
                                                  allow_ms_adjacent_slot_ramp_bypass: bool) -> bool:
        """
        This function handles looking for and configuring preceding bursts already scheduled in `_queue` that would
        be compatible for being continuous with subsequent planned scheduled bursts


        :param carrier: RFCarrier that the planned scheduled burst(s) will use, used by function to determine if
         preceding bursts whose carrier matches the passed one.
        :type carrier: RFCarrier
        :param start_time: The start time of the desired scheduled burst, the function checks for preceding scheduled
         bursts that end at `start_time` on the same `carrier` and thus are contiguous
        :type start_time: TDMATime
        :param burst_type: The type of burst being scheduled for which the function needs to identify and attempt to
         make any preceding scheduled burst in `_queue` continuous with if their types are compatible
        :type burst_type: type[Burst]
        :param allow_ms_adjacent_slot_ramp_bypass: A bool, True if the MS/transmitter is in multi-slot transmission mode
         and wishes to not ramp up between contiguous MS Normal Uplink Bursts, False is otherwise.
        :type allow_ms_adjacent_slot_ramp_bypass: bool
        :return: Returns True if the function determines and then sucessfully configures a preceding contiguous
         scheduled block in `_queue` to be continuous with the planned subsequent block, False otherwise
        :rtype: bool
        """
        # 1. Check if we allow for continuous bursts with our current burst_type before anything else
        if burst_type.CONTINUITY_MODE == BurstContinuity.ISOLATED or (
         burst_type.CONTINUITY_MODE == BurstContinuity.OPTIONAL and not allow_ms_adjacent_slot_ramp_bypass):
            return False
        # 2. Determine if there is a preceding burst that is adjacent in end_time to our burst and has same carrier
        lower = bisect_left(self._queue, (start_time - TIMESLOT_SUBSLOT_LENGTH), key=lambda x: x.start_time)
        upper = bisect_left(self._queue, start_time, key=lambda x: x.start_time)
        preceding_block = None
        for block in self._queue[lower:upper]:
            # block(s) are within the preceding timeslot
            if block.end_time == start_time and block.carrier.id == carrier.id:
                # block ends is contiguous and shares rf carrier
                preceding_block = block
                break

        if preceding_block is None:
            return False
        # Check if the adjacent burst is compatible and or allows for continuous bursts
        compatible = (burst_type.CONTINUITY_BURST_TYPE in
                      preceding_block.burst.CONTINUITY_COMPATIBLE_BURST_TYPES)

        # 3. If compatible, revise the preceding contiguous burst to be continuous
        match preceding_block.burst.CONTINUITY_MODE:
            case BurstContinuity.REQUIRED:
                if compatible:
                    return self._revise_burst_ramping_to_continuous(preceding_block)
            case BurstContinuity.OPTIONAL:
                if allow_ms_adjacent_slot_ramp_bypass:
                    if compatible:
                        return self._revise_burst_ramping_to_continuous(preceding_block)
            case BurstContinuity.ISOLATED:
                pass

        return False

    def _construct_burst_block_list(self, burst_type: type[Burst],
                                    input_logical_ch: tuple[list[LogicalChannelVD | None], ...],
                                    phy_channel: PhysicalChannel | None,
                                    start_time: TDMATime | None, *,
                                    allow_ms_adjacent_slot_ramp_bypass: bool,
                                    continuous_with_prior_blocks: bool,
                                    forced_scheduling: bool,
                                    fill_empty_channels: bool) -> list[BurstBlock] | None:
        """
        Refer to documentation for `schedule_bursts`, the primary difference is that the function wraps this one
        and inserts the generated BurstBlocks in the `_queue` while this one simply returns in a list.
        """

        # 1. Handle processing input logical channels and blocks to determine total length
        local_logical_ch_input: tuple[list[LogicalChannelVD | None], ...] = tuple(list(x) for x in input_logical_ch)
        stream_lengths = [0] * len(input_logical_ch)

        # Remove "trailing" None's, only preceding Nones are useful in determining output length
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

        # 2. Handle physical channel and timing input parameters
        if phy_channel is None:
            if not forced_scheduling:
                raise ValueError("BurstStreamBuilder was passed None for phy_channel, but forced=False, invalid"
                                 " input argument combination, expected specified phy_channel or forced=True")
            phy_channel = TETRA_DEFAULT_PHY_DICT[burst_type.DEFAULT_PHY]

        if start_time is None:
            start_time = self._next_available_time_for_carrier(phy_channel.carrier, burst_type.subslot_width)
        else:
            if start_time > self.current_tetra_time:
                raise ValueError(f"Passed time to BurstStreamBuilder is in past: {start_time}"
                                 f" compared to current Builder time: {self.current_tetra_time}")
            end_time = start_time + ((output_length - 1) * TIMESLOT_SUBSLOT_LENGTH) + burst_type.subslot_width
            collision = self._check_scheduled_bursts_for_collisions(phy_channel.carrier, start_time, end_time)
            if collision:
                raise ScheduledBurstCollisionError("Cannot schedule burst(s) on carrier: "
                                                   f"{phy_channel.carrier}"
                                                   f", starting at: {start_time} and ending at: {end_time}")
        # print(f"Determined start time: {start_time}")
        # 3. Sort through logical channels to allow for easier use in burst creation argument unpacking
        logical_chs_list: list[list[LogicalChannelVD]] = [[] for _ in range(output_length)]
        logical_ch_views_list: list[list[LogicalChannelVD]] = [[] for _ in range(output_length)]
        logical_ch_block_indices_list: list[list[int]] = [[] for _ in range(output_length)]

        num_input_channels = len(local_logical_ch_input)
        for i in range(num_input_channels):
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
            block_1_cont_with_prior = self._handle_prior_contiguous_burst_continuity(phy_channel.carrier,
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
            burst = burst_type(phy_channel=phy_channel, tetra_time=sched_time, forced=forced_scheduling)
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

    def get_scheduled_bursts(self, time: TDMATime | None = None, number_of_full_timeslots: int = 1,
                             end_time: TDMATime | None = None,
                             increment_time_on_call: bool = True,
                             return_all_future_bursts: bool = False) -> list[BurstBlock]:
        """
        This function returns all scheduled bursts in `_queue` that have start between `start_time` and some end time
        but end before the end time. It can also return all scheduled bursts after some specified time with
        `return_all_future_bursts` argument, and by specifying no arguments, will operate in a default self-incrementing
        mode that will return all bursts in a timeslot, then advance to the next timeslot and so on and so forth

        **NOTE 1:** either specify `end_time` and leave `number_of_full_timeslots` default, or leave `end_time` None and
         specify `number_of_full_timeslots`, do not specify both.

        **NOTE 2:** Because BurstStreamBuilder can schedule bursts at the same time with unique RFCarriers /
        PhysicalChannels, the return list can be empty if no bursts that are scheduled match arguments, or it can be
        longer than expected because different carriers, and 2 subslot bursts can occupy a single timeslot.

        **NOTE 3:** The return list of BurstBlocks is order by start time, and for any bursts that have the same start
        time, is ordered secondily by the `channel_number` of the RFCarrier for the burst's PhysicalChannel.

        :param time: The TDMAtime for which to get any scheduled burst(s) that start at that time, if None,
         assumes the `current_tetra_time` of the BurstStreamBuilder instance instead
        :type time: TDMATime | None, default None
        :param number_of_ful_timeslots: The number of timeslots for which to get bursts out, default behaviour is 1,
         in that the function returns all bursts scheduled from start_time to before the next timeslot
        :type number_of_full_timeslots: int, default 1
        :param end_time: Alternative method of specifying duration for which to get bursts from, will return all bursts
         scheduled from start_time and have end time at end_time, refer to **NOTE** above for use of `end_time`
        :type end_time: TDMATime | None, default None
        :param increment_time_on_call: A bool, if True, function will increment `current_tetra_time` of the object by
         the total time lapsed by the return BurstBlocks and will delete all returned bursts from `_queue`
        :type increment_time_on_call: bool. default True
        :param return_all_future_bursts: A bool, if True, function will return all bursts that are scheduled after the
         passed time
        :type return_all_future_bursts: bool. default False
        :return: Returns an ordered list in time, and RFCarrier channel number for ties, of the matching scheduled
         BurstBlocks in `_queue` of the BurstStreamBuilder
        :rtype: List[BurstBlock]
        """
        # 1. Handle `time` and `end_time` arguments
        if time is None:
            time = self.current_tetra_time
        else:
            if time < self.current_tetra_time:
                raise ValueError(f"Passed `time` for `get_scheduled_bursts` is in the past: {time}, compared to"
                                 f" current `BurstStreamBuilder` time: {self.current_tetra_time}")

        if end_time is not None and number_of_full_timeslots != 1:
            raise ValueError("Specified `end_time` argument and `number_of_time_slots` to `get_scheduled_bursts`,"
                             " invalid argument combo, specify either `number_of_timeslots` or `end_time`, not both")

        if end_time is None:
            end_time = time.copy()
            end_time.advance_to_next_timeslot()
            end_time += (number_of_full_timeslots-1) * TIMESLOT_SUBSLOT_LENGTH
        else:
            if end_time <= time:
                raise ValueError(f"Passed `end_time` for `get_scheduled_bursts`: {end_time}, is <= compared to"
                                 f" the passed start `time`: {time}")

        # 2. Determine indices of `self._queue` to return
        get_start_index = bisect_left(self._queue, time, key=lambda x: x.start_time)

        if return_all_future_bursts:
            get_end_index = len(self._queue)
        else:
            get_end_index = bisect_left(self._queue, end_time, key=lambda x: x.start_time)

        return_burst_list = self._queue[get_start_index:get_end_index]

        if increment_time_on_call:
            del self._queue[get_start_index:get_end_index]
            self.current_tetra_time = end_time

        return return_burst_list

    def schedule_bursts(self, burst_type: type[Burst],
                        input_logical_ch: tuple[list[LogicalChannelVD | None], ...],
                        phy_channel: PhysicalChannel | None = None,
                        start_time: TDMATime | None = None, *,
                        allow_ms_adjacent_slot_ramp_bypass: bool = False,
                        continuous_with_prior_blocks: bool = True,
                        forced_scheduling: bool = False,
                        fill_empty_channels: bool = False) -> None:
        """
        This function attempts to schedule the specified burst(s) based on the passed input parameters.

        **NOTE 1:** Multi-burst scheduling
            - `schedule_bursts` supports passing a `input_logical_ch` tuple that will result in multiple bursts to be
             scheduled.
            - Recall that logical channel types are seperated into *traffic* and *control* types, *traffic*
             types support m>1, that is multiple burst blocks. That occurs when interleaving is done and or multiple
             input blocks are passed to them, because interleaving of the input is needed, these types are required
             to support multiple output blocks.
            - Recall that `burst_types` consist of 8 types, and support between 1-3 input logical channels, many of
             which support alternative configuration and not a fixed value of 1,2, or 3.
            - The result of the above facts is that `input_logical_ch` is the most *complicated or obscure* argument,
             to help an example is given below


            - **EX** say we want to schedule 5 bursts of type NormalContinousDownlink type, that type can accept up
                to 3 logical channel inputs: bkn1 (block 1), bbk (broadcast block), and bkn2 (block 2), typically
                we pass 2 input channels a traffic and access assignment control channel (AACH), traffic is used to
                fill up bkn1 and bkn2 while bbk is for AACH,
            - However, in some cases the control plane needs to steal traffic blocks for immediate control signaling
                thus bkn2 is actually set to be filled by a STCH (stealing channel) logical input block
            - To describe these 5 bursts the follow would be the `input_logical_ch` parameter:
                `input_logical_ch=([1_traffic(4), 5_traffic(1)], [1_ACCH(1), 2_ACCH(1), ..., 5_ACCH(m=1)], [None, None,
                None, None, 5_STCH(1)])`
            - In this case the None's are padding in the third tuple element to indicate that there is no 3rd logical
                input channel used for the first 4 bursts, and then on the 5th burst a STCH channel is used for a 3rd
                ch input.
            - Also note that because AACH is control type channel it does not support (m>1) and thus 5 seperate ones
                had to be passed, extrawork can be eased by using `fill_empty_channels=True` to fill passed empty ch's

        :param burst_type: Burst subclass type, NormalUplinkBurst, ControlUplink, SyncContDownlinkBurst,
         NormalDiscontDownlinkBurst, etc. **NOTE**: not a instance but simply the class/type desired to be scheduled.
        :type burst_type: type[Burst]
        :param input_logical_ch: A tuple of logical_channel blocks that the `burst_type` requires, in some cases a
         `burst_type` may require 1 to 3 blocks, futher more, if the caller desired to schedule multiple bursts,
         each of the (1-3) tuple values is a multi-element list or a traffic-type logical channel where m>1 in the cases
         of interleaving, **NOTE** refer to remarks above regarding multiburst scheduling for more information.
        :type input_logical_ch: Tuple[List[LogicalChannelVD | None], ...]
        :param phy_channel: The PhysicalChannel
        :type phy_channel: PhysicalChannel | None, default None
        :param start_time: The start time of the burst(s) scheduled, note that when `input_logical_ch` passed is setup
         to fill more than one bursts, any subsequent burst is scheduled after the first burst contiguously untill all
         bursts are scheduled. **NOTE**: If None, BurstStreamBuilder attempts to find the earliest subsequent time
         to scheduled the burst(s) after any already scheduled bursts with the same `RFCarrier`, if no other bursts
         with the same carrier are scheduled, returns schedules the start time of the burst(s) to be at
         `current_tetra_time`
        :type start_time: TDMATime | None, default None
        :param allow_ms_adjacent_slot_ramp_bypass: Boolean, True if an MS has support for multi-slot uplink transmission
         ,that is transmitting on adjacent timeslots, and wishes to not ramp power up and down between subsequent
         timeslots, default False, **NOTE**: only affects NormalUplinkBursts that scheduled consecutively
        :type allow_ms_adjacent_slot_ramp_bypass: bool, default False
        :param continuous_with_prior_blocks: Boolean, True if the caller wants allow for checking and reconfiguration
         of already scheduled compatible preceding bursts to be continuous with the current bursts, **NOTE**: only
         applies to bursts that support continuous bursts (no ramping up/down between timeslots) and ones whoose burst
         types are intercompatible.
        :type continuous_with_prior_blocks: bool, default True
        :param forced_scheduling: Boolean, when True tell's generated bursts to ignore any incompatibility of the
         scheduled burst in terms of time and burst type. **NOTE**: does not ignore incompatibility of bursts, time, and
         physical channel all together, therefore passed physical_channels must be correct for the given burst and
         logical channels.
        :type forced_scheduling: bool, default False
        :param fill_empty_channels: Boolean, if True BurstStreamBuilder will fill any passed logical channel in
          `input_logical_ch` that does not have type 5 blocks generated using the logical channel methods and configured
          seed with random data, **NOTE**: this does not affect the passed logical channels, a local copy is made and
          stored within the scheduled `BurstBlock`'s
        :type fill_empty_channels: bool, default False
        """

        # 1. Generate burst block list
        blocks = self._construct_burst_block_list(burst_type, input_logical_ch, phy_channel, start_time,
                                                  allow_ms_adjacent_slot_ramp_bypass=allow_ms_adjacent_slot_ramp_bypass,
                                                  continuous_with_prior_blocks=continuous_with_prior_blocks,
                                                  forced_scheduling=forced_scheduling,
                                                  fill_empty_channels=fill_empty_channels)
        if blocks is not None:
            # 2. Insert blocks into queue
            insert_index = bisect_left(self._queue, blocks[0])
            self._queue[insert_index:insert_index] = blocks
###################################################################################################
