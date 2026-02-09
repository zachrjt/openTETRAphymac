"""
physical_channels.py contains the implementations for each type of valid uplink and downlink burst for TETRA V2.4.2.
each burst has in-depth validation methods to ensure only permissible bursts are being constructed based on subslot,
slot, frame, multiframe numbers and physical channels.

Each burst contains a construct_burst_sequence method which takes in 1-3 logical channels, depedant on the type of burst
, validates that the input logical channels are valid and a burst can be built with the passed Physical Channel object
during initilization and the current subslot, slot, frame, multiframe numbers.

It then constructs the 510 modulation bits for the burst using the logical channel data and training sequences,
phase adjustment bits, etc.

The returned modulation bits are ready to be fed into a RFTransmitter class alongside the start_ramp_period and
end_ramp_periods.
"""

from typing import ClassVar, Protocol
from dataclasses import dataclass

from numpy import uint8, array, empty, zeros
from numpy.typing import NDArray

from .constants import PhyType, LinkDirection, BurstContent, ChannelKind, ChannelName
from .logical_channels import BLCH, BNCH, BSCH, CLCH, SCH_HD, SCH_F, SCH_HU, AACH, LogicalChannelVD, \
    TrafficChannel, STCH
from .modulation import calculate_phase_adjustment_bits

# Following constants Per EN 300 392-2 V2.4.2 - 9.3
HYPERFRAME_MULTIFRAME_LENGTH = 60   # Number of multiframes in a hyperframe
MULTIFRAME_TDMAFRAME_LENGTH = 18    # Number of frames in a multiframe
CONTROL_FRAME_NUMBER = 18           # The frame number (FN) of the control frame (last frame in multiframe)
TDMAFRAME_TIMESLOT_LENGTH = 4       # How many timeslots in a frame
TIMESLOT_BIT_LENGTH = 510           # How many modulation bits in a timeslot
TIMESLOT_SUBSLOT_LENGTH = 2         # How many subslots in a timeslot

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.1
FREQUENCY_CORRECTION_FIELD = array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1, 1], dtype=uint8)

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.2
NORMAL_TRAINING_SEQUENCE = array([[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
                                  [0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
                                  [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]], dtype=uint8)

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.3
EXTENDED_TRAINING_SEQUENCE = array(
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=uint8)

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.4
SYNCHRONIZATION_TRAINING_SEQUENCE = array(
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
     0, 0, 0, 1, 1, 0, 0, 1, 1, 1], dtype=uint8)

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.5
TAIL_BITS = array([1, 1, 0, 0], dtype=uint8)

# Per EN 300 392-2 V2.4.2 - 9.4.4.3.6
PHASE_ADJUSTMENT_SYMBOL_RANGE = {"a": (7, 121), "b": (122, 248), "c": (7, 107), "d": (108, 248), "e": (111, 229),
                                 "f": (0, 110), "g": (2, 116), "h": (117, 243), "i": (2, 102), "j": (103, 243)}


###################################################################################################
@dataclass(frozen=True, slots=True)
class PhysicalChannel():
    channel_number: int          # The channel number ""
    main_carrier: bool           # If the carrier used is considered the "mainCarrier"
    ul_frequency: float         # The UL frequency for MS->BS tx
    dl_frequency: float         # The DL frequency for BS->MS tx
    channel_type: PhyType        # The type of physical channel (CP,TP,UP)
###################################################################################################


class Burst():
    # require class-level "spec constants"
    sn_max: ClassVar[int]                   # Max number of modulation symbols in burst
    start_guard_bit_period: ClassVar[int]     # The initial guard period / delay in bits (default)
    end_guard_bit_period: ClassVar[int]       # The end guard period in bits (default)
    subslot_width: ClassVar[int]                # How many subslots does the burst take up: 1 or 2
    link_direction: ClassVar[LinkDirection]     # either DL or UL

    # Permissible physical channel types for the burst class
    ALLOWED_PHY: ClassVar[set[PhyType]]

    # per instance runtime variables
    burst_type: BurstContent        # either traffic, control, or mixed
    phy_channel: PhyType            # The physical RF channel type (CP,TP,UP) that the burst utilizes
    mixed_burst: bool               # if the burst has 1 or more blocks/subslots stolen for control or a composite burst
    multiframe_number: int          # The multi-frame number of the burst
    frame_number: int               # The frame number of the burst
    timeslot: int                   # The TDMA time slot number of the burst
    subslot: int                    # The subslot which the burst starts in and occupies atleast

    # Variable versions of start/end_guard_bit_period
    # that allow for dynamic ramping based on if we have continuous data
    start_ramp_period: int
    end_ramp_period: int

    def __init__(self, phy_channel: PhysicalChannel, mn: int, fn: int, tn: int, ssn: int = 1):
        self._validate_class_constants()
        # store runtime state
        self.phy_channel = phy_channel.channel_type
        self.multiframe_number = mn
        self.frame_number = fn
        self.timeslot = tn
        self.subslot = ssn

        # default; burst building may change later on to True
        self.mixed_burst = False
        # default is the guard delay periods, but burst building may change later on
        self.start_ramp_period = self.start_guard_bit_period
        self.end_ramp_period = self.end_guard_bit_period

        self._validate_common()

    def _validate_common(self) -> None:
        # TN range checks
        if not 1 <= self.timeslot <= TDMAFRAME_TIMESLOT_LENGTH:
            raise ValueError(f"TN {self.timeslot} invalid for {type(self).__name__}")
        # FN range checks
        if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
            raise ValueError(f"FN {self.frame_number} invalid for {type(self).__name__}")
        # MN range checks
        if not 1 <= self.multiframe_number <= HYPERFRAME_MULTIFRAME_LENGTH:
            raise ValueError(f"MN {self.multiframe_number} invalid for {type(self).__name__}")
        # SSN range checks depending on subslot_width
        if not 1 <= self.subslot <= TIMESLOT_SUBSLOT_LENGTH:
            raise ValueError(f"SSN {self.subslot} invalid for {type(self).__name__}")
        # Physical channel type allowed
        if self.phy_channel not in self.ALLOWED_PHY:
            raise ValueError(f"Phy {self.phy_channel} invalid for {type(self).__name__}")

    def _validate_class_constants(self) -> None:
        required_vars = ["sn_max", "start_guard_bit_period", "end_guard_bit_period",
                         "subslot_width", "link_direction", "ALLOWED_PHY"]
        for name in required_vars:
            if not hasattr(type(self), name):
                raise TypeError(f"{type(self).__name__} missing class constant {name}")

###################################################################################################


class ControlUplink(Burst):
    sn_max = 103
    start_guard_bit_period = 34
    end_guard_bit_period = 15
    subslot_width = 1
    link_direction = LinkDirection.UPLINK
    burst_type = BurstContent.BURST_CONTROL_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL}

    def construct_burst_sequence(self, input_logical_ch_ssn1: SCH_HU) -> NDArray[uint8]:
        # Must verify specific non-common TN/FN based on physical channel
        if self.phy_channel == PhyType.CONTROL_CHANNEL:
            if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}, FN {self.frame_number}"
                                 f" invalid for ssn{self.subslot}: {input_logical_ch_ssn1.channel}")
        elif self.phy_channel == PhyType.TRAFFIC_CHANNEL:
            if self.frame_number != CONTROL_FRAME_NUMBER:
                raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}, FN {self.frame_number}"
                                 f" invalid for ssn{self.subslot}: {input_logical_ch_ssn1.channel}")
        # runtime check to verify channel type
        if input_logical_ch_ssn1.channel != ChannelName.SCH_HU_CHANNEL:
            raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel},"
                             f" invalid ssn of {input_logical_ch_ssn1.channel}")
        # Build the burst
        d = self.start_guard_bit_period
        burst_bit_seq = empty(shape=(self.sn_max*2)+d+self.end_guard_bit_period, dtype=uint8)
        burst_bit_seq[:d] = zeros(shape=self.start_guard_bit_period, dtype=uint8)   # guard period
        burst_bit_seq[d:4+d] = TAIL_BITS
        burst_bit_seq[d+4:88+d] = input_logical_ch_ssn1.type_5_blocks[0][:84]
        burst_bit_seq[d+88:118+d] = EXTENDED_TRAINING_SEQUENCE
        burst_bit_seq[d+118:202+d] = input_logical_ch_ssn1.type_5_blocks[0][84:168]
        burst_bit_seq[d+202:206+d] = TAIL_BITS
        burst_bit_seq[206+d:255] = zeros(shape=self.end_guard_bit_period, dtype=uint8)  # guard period
        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:

        raise NotImplementedError

###################################################################################################


class NormalUplinkBurst(Burst):

    sn_max = 231
    start_guard_bit_period = 34
    end_guard_bit_period = 14
    subslot_width = 2
    link_direction = LinkDirection.UPLINK
    burst_type = BurstContent.BURST_MIXED_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL}

    def construct_burst_sequence(self, input_logical_ch_bkn1: TrafficChannel | SCH_F | STCH,
                                 input_logical_ch_bkn2: TrafficChannel | STCH | None = None,
                                 ramp_up_down_state: tuple[bool, bool] = (True, True)) -> NDArray[uint8]:

        bkn1 = input_logical_ch_bkn1.channel_type
        bkn2 = None if input_logical_ch_bkn2 is None else input_logical_ch_bkn2.channel_type
        if input_logical_ch_bkn1.channel not in (ChannelName.TCH_2_4_CHANNEL, ChannelName.TCH_4_8_CHANNEL,
                                                 ChannelName.TCH_7_2_CHANNEL, ChannelName.TCH_S_CHANNEL,
                                                 ChannelName.SCH_F_CHANNEL, ChannelName.STCH_CHANNEL):
            raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                             f" invalid bkn1:{input_logical_ch_bkn1.channel}")
        if (input_logical_ch_bkn2 is not None
            and input_logical_ch_bkn2.channel not in (ChannelName.TCH_2_4_CHANNEL, ChannelName.TCH_4_8_CHANNEL,
                                                      ChannelName.TCH_7_2_CHANNEL, ChannelName.TCH_S_CHANNEL,
                                                      ChannelName.STCH_CHANNEL)):

            raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                             f" invalid bkn2:{input_logical_ch_bkn2.channel}")
        training_seq_index = 0  # Depending on composition of burst, we use a different training sequence either 1 or 2

        match (bkn1, bkn2):
            case (ChannelKind.TRAFFIC_TYPE, None):
                # Pure TCH on TP with FN:[1,17]
                self.mixed_burst = False
                self.burst_type = BurstContent.BURST_TRAFFIC_TYPE
                training_seq_index = 0
                if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f", FN {self.frame_number} invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, None):
                # Pure SCH/F on CP with FN:[1,18] or on TP with FN:18
                self.mixed_burst = False
                self.burst_type = BurstContent.BURST_CONTROL_TYPE
                training_seq_index = 0
                if self.phy_channel == PhyType.CONTROL_CHANNEL:
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                         f", FN {self.frame_number} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                elif self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                         f", FN {self.frame_number} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    raise ValueError(f"For {self.__class__.__name__}, phy {self.phy_channel}"
                                     f", invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, ChannelKind.TRAFFIC_TYPE):
                # BKN1 stolen for STCH on TP with FN:[1,17], and BKN2 as TCH
                self.mixed_burst = True
                self.burst_type = BurstContent.BURST_TRAFFIC_TYPE
                training_seq_index = 1
                if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f", FN {self.frame_number} invalid for bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelKind.CONTROL_TYPE, _):
                if bkn2 == ChannelKind.CONTROL_TYPE:
                    if input_logical_ch_bkn2 is not None:
                        if input_logical_ch_bkn2.channel != input_logical_ch_bkn1.channel:
                            raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                             f" invalid combination bkn1:{bkn1} bkn2:{bkn2}")

                    # BKN1 and BKN2 stolen for STCH on TP with FN:[1,17]
                    self.mixed_burst = True
                    self.burst_type = BurstContent.BURST_TRAFFIC_TYPE
                    training_seq_index = 1
                    if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                         f" invalid for bkn1:{bkn1} bkn2:{bkn2} (expected TP)")
                    if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                         f", FN {self.frame_number} invalid for bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    # Invalid combo of channels
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
            case _:
                # Invalid combo of channels
                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

        # Build the burst
        d = self.start_guard_bit_period
        burst_bit_seq = empty(shape=(self.sn_max*2)+d+self.end_guard_bit_period, dtype=uint8)

        burst_bit_seq[d:4+d] = TAIL_BITS
        burst_bit_seq[d+4:220+d] = input_logical_ch_bkn1.type_5_blocks[0][:216]
        burst_bit_seq[d+220:242+d] = NORMAL_TRAINING_SEQUENCE[training_seq_index]
        if input_logical_ch_bkn2 is not None and self.mixed_burst:
            burst_bit_seq[d+242:458+d] = input_logical_ch_bkn2.type_5_blocks[0][:216]
        else:
            burst_bit_seq[d+242:458+d] = input_logical_ch_bkn1.type_5_blocks[0][216:432]
        burst_bit_seq[d+458:462+d] = TAIL_BITS

        # must add guard period training sequence if there is no ramping at the start
        if ramp_up_down_state[0]:
            burst_bit_seq[:d] = zeros(shape=self.start_guard_bit_period, dtype=uint8)
            self.start_ramp_period = d
        else:
            # add preceding bits per 9.4.5.3
            burst_bit_seq[:30] = EXTENDED_TRAINING_SEQUENCE
            burst_bit_seq[30:32] = TAIL_BITS[2:]
            # Insert phase adjustment bits f
            burst_bit_seq[32:d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                  PHASE_ADJUSTMENT_SYMBOL_RANGE["f"], d)
            self.start_ramp_period = 0
        # must add guard period training sequence if there is no ramping at the end
        if ramp_up_down_state[1]:
            burst_bit_seq[462+d:510] = zeros(shape=self.end_guard_bit_period, dtype=uint8)
            self.end_ramp_period = TIMESLOT_BIT_LENGTH - (462+d)
        else:
            # add following bits per 9.4.5.3
            # Insert phase adjustment bits f
            burst_bit_seq[462+d:464+d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                         PHASE_ADJUSTMENT_SYMBOL_RANGE["e"], d)
            burst_bit_seq[464+d:466+d] = TAIL_BITS[:2]
            burst_bit_seq[466+d:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.end_ramp_period = 0

        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:

        raise NotImplementedError

###################################################################################################


class DownlinkHost(Protocol):
    # host protocol for the mixin to satisfy typing for both normal and synchronous downlink mixins
    phy_channel: PhyType
    frame_number: int
    multiframe_number: int
    timeslot: int


class NormalDownlinkMixin:

    def _norm_bkin(self, logical_ch: LogicalChannelVD) -> str:
        return (logical_ch.channel_type if logical_ch.channel_type == ChannelKind.TRAFFIC_TYPE
                else logical_ch.channel)

    def _validate_normal_downlink_mapping(self: DownlinkHost, bkn1: str,
                                          bkn2: str | None) -> tuple[BurstContent, bool, int]:
        """
        Validates (bkn1, bkn2) for normal downlink burst usage across CP/TP/UP
        and returns:
            (burst_type: BurstContent, mixedBurst: bool, training_seq_index: int)
        """
        if bkn2 is None:
            if bkn1 not in (ChannelKind.TRAFFIC_TYPE, ChannelName.SCH_F_CHANNEL):
                raise ValueError(
                    f"For {type(self).__name__}, bkn2 is None but bkn1:{bkn1} is not in "
                    f"({ChannelKind.TRAFFIC_TYPE}, {ChannelName.SCH_F_CHANNEL})"
                    )

        match (bkn1, bkn2):
            case (ChannelKind.TRAFFIC_TYPE, None):
                # Pure TCH on TP with FN:[1,17]
                if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phy_channel} invalid for pure TCH (expected TP). "
                        f"bkn1:{bkn1} bkn2:{bkn2}")
                if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                    raise ValueError(
                        f"For {type(self).__name__}, FN {self.frame_number} invalid for pure TCH on TP "
                        f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1}).")
                return (BurstContent.BURST_TRAFFIC_TYPE, False, 0)

            case (ChannelName.SCH_F_CHANNEL, None):
                # Pure SCH/F on CP with FN:[1,18] or on TP with FN:18
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/F on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER}).")
                elif self.phy_channel == PhyType.CONTROL_CHANNEL:
                    if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/F on CP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1}).")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, False, 0)

            case (ChannelName.STCH_CHANNEL, ChannelKind.TRAFFIC_TYPE):
                # BKN1 stolen for STCH on TP FN:[1,17], BKN2 as TCH
                if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phy_channel} invalid for STCH+TCH (expected TP). "
                        f"bkn1:{bkn1} bkn2:{bkn2}")
                if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                    raise ValueError(
                        f"For {type(self).__name__}, FN {self.frame_number} invalid for STCH+TCH on TP "
                        f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1}).")
                return (BurstContent.BURST_MIXED_TYPE, True, 1)

            case (ChannelName.STCH_CHANNEL, _):
                if bkn2 == ChannelName.STCH_CHANNEL:
                    # BKN1 and BKN2 stolen for STCH on TP FN:[1,17]
                    if self.phy_channel != PhyType.TRAFFIC_CHANNEL:
                        raise ValueError(
                            f"For {type(self).__name__}, phy {self.phy_channel} invalid for STCH+STCH (expected TP). "
                            f"bkn1:{bkn1} bkn2:{bkn2}"
                        )
                    if not 1 <= self.frame_number < MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for STCH+STCH on TP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH-1}).")
                    return (BurstContent.BURST_MIXED_TYPE, True, 1)

                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL):
                # SCH/HD + BNCH on TP FN:18 or CP FN:[1,18]
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+BNCH on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER}).")
                    if ((self.multiframe_number + self.timeslot) % 4) != 1:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 1 for SCH/HD+BNCH on TP "
                            f"(MN={self.multiframe_number}, TN={self.timeslot}).")
                elif self.phy_channel == PhyType.CONTROL_CHANNEL:
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+BNCH on CP "
                            f"(expected 1..{MULTIFRAME_TDMAFRAME_LENGTH}).")
                    if self.frame_number == CONTROL_FRAME_NUMBER:
                        if ((self.multiframe_number + self.timeslot) % 4) != 1:
                            raise ValueError(
                                f"For {type(self).__name__}, (MN+TN)%4 != 1 for SCH/HD+BNCH on CP control frame "
                                f"(MN={self.multiframe_number}, TN={self.timeslot}).")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, True, 1)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BLCH_CHANNEL):
                # SCH/HD + BLCH on TP FN:18 or CP/UP FN:[1,18]
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+BLCH on TP "
                            f"(expected control frame {CONTROL_FRAME_NUMBER}).")
                elif self.phy_channel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+BLCH on "
                            f"{self.phy_channel} (expected 1..{MULTIFRAME_TDMAFRAME_LENGTH}).")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (BurstContent.BURST_CONTROL_TYPE, True, 1)

            case (ChannelName.SCH_HD_CHANNEL, _):
                if bkn2 == ChannelName.SCH_HD_CHANNEL:
                    # SCH/HD + SCH/HD on TP FN:18 or CP/UP FN:[1,18]
                    if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                        if self.frame_number != CONTROL_FRAME_NUMBER:
                            raise ValueError(
                                f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+SCH/HD on TP "
                                f"(expected control frame {CONTROL_FRAME_NUMBER})."
                            )
                    elif self.phy_channel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                        if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                            raise ValueError(
                                f"For {type(self).__name__}, FN {self.frame_number} invalid for SCH/HD+SCH/HD on "
                                f"{self.phy_channel} (expected 1..{MULTIFRAME_TDMAFRAME_LENGTH}).")
                    else:
                        raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                         f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                    return (BurstContent.BURST_CONTROL_TYPE, True, 1)

                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

            case _:
                raise ValueError(f"For {type(self).__name__}, invalid combination of {bkn1} and {bkn2}")

###################################################################################################


class NormalContDownlinkBurst(NormalDownlinkMixin, DownlinkHost, Burst):
    sn_max = 255
    start_guard_bit_period = 0
    end_guard_bit_period = 0
    subslot_width = 2
    link_direction = LinkDirection.DOWNLINK
    burst_type = BurstContent.BURST_TRAFFIC_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    phy_channel: PhyType
    frame_number: int
    multiframe_number: int
    timeslot: int

    def construct_burst_sequence(self, input_logical_ch_bkn1: TrafficChannel | SCH_F | STCH | SCH_HD,
                                 input_logical_ch_bbk: AACH,
                                 input_logical_ch_bkn2: TrafficChannel | BLCH | BNCH | STCH | SCH_HD | None = None,
                                 ramp_up_down_state: tuple[bool, bool] = (False, False)) -> NDArray[uint8]:
        # If bkn1 is control, we care about the specific channel type, if it is traffic we dont care for bkn1
        bkn1 = self._norm_bkin(input_logical_ch_bkn1)
        bkn2 = None if input_logical_ch_bkn2 is None else self._norm_bkin(input_logical_ch_bkn2)

        if input_logical_ch_bbk.channel != ChannelName.AACH_CHANNEL:
            raise ValueError(f"Passed broadcast block is invalid for {type(self).__name__} expected"
                             f" {ChannelName.AACH_CHANNEL}")

        burst_type, multiple_logical_ch_state, tsi = self._validate_normal_downlink_mapping(bkn1, bkn2)
        self.burst_type = burst_type
        self.mixed_burst = multiple_logical_ch_state
        training_seq_index = tsi

        # Build the burst
        burst_bit_seq = empty(shape=(self.sn_max*2), dtype=uint8)
        if ramp_up_down_state[0]:
            # if we are ramp up (TRUE), it means that this is the first burst
            burst_bit_seq[:12] = zeros(shape=12, dtype=uint8)
            self.start_ramp_period = 12
        else:
            # other we are continuous (or we are ramping down add preceding bits per 9.4.5.1 - Table 28)
            burst_bit_seq[:12] = NORMAL_TRAINING_SEQUENCE[2][10:22]
            self.start_ramp_period = 0

        # temporarily skip phase adjustment bits a - [12:14]
        burst_bit_seq[14:230] = input_logical_ch_bkn1.type_5_blocks[0][:216]
        burst_bit_seq[230:244] = input_logical_ch_bbk.type_5_blocks[0][:14]
        burst_bit_seq[244:266] = NORMAL_TRAINING_SEQUENCE[training_seq_index][:22]
        burst_bit_seq[266:282] = input_logical_ch_bbk.type_5_blocks[0][14:30]
        if input_logical_ch_bkn2 is not None and self.mixed_burst:
            burst_bit_seq[282:498] = input_logical_ch_bkn2.type_5_blocks[0][:216]
        else:
            burst_bit_seq[282:498] = input_logical_ch_bkn1.type_5_blocks[0][216:432]
        # temporarily skip phase adjustment bits b - [498:500]
        if ramp_up_down_state[1]:
            # if we are ramp down (TRUE), it means that this is the last burst we are ramping down
            burst_bit_seq[500:510] = zeros(shape=10, dtype=uint8)
            self.end_ramp_period = 10
        else:
            # otherwise we are continuous (or we are have ramped up add preceding bits per 9.4.5.1 - Table 27)
            burst_bit_seq[500:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.end_ramp_period = 0

        # Now insert phase adjustment bits
        burst_bit_seq[12:14] = calculate_phase_adjustment_bits(burst_bit_seq, PHASE_ADJUSTMENT_SYMBOL_RANGE['a'], 0)
        burst_bit_seq[498:500] = calculate_phase_adjustment_bits(burst_bit_seq, PHASE_ADJUSTMENT_SYMBOL_RANGE['b'], 0)

        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:
        raise NotImplementedError

###################################################################################################


class NormalDiscontDownlinkBurst(NormalDownlinkMixin, DownlinkHost, Burst):
    sn_max = 246
    start_guard_bit_period = 10
    end_guard_bit_period = 8
    subslot_width = 2
    link_direction = LinkDirection.DOWNLINK
    burst_type = BurstContent.BURST_TRAFFIC_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    phy_channel: PhyType
    frame_number: int
    multiframe_number: int
    timeslot: int

    def construct_burst_sequence(self, input_logical_ch_bkn1: TrafficChannel | SCH_F | STCH | SCH_HD,
                                 input_logical_ch_bbk: AACH,
                                 input_logical_ch_bkn2: TrafficChannel | BLCH | BNCH | STCH | SCH_HD | None = None,
                                 ramp_up_down_state: tuple[bool, bool] = (True, True)) -> NDArray[uint8]:
        # If bkn1 is control, we care about the specific channel type, if it is traffic we dont care for bkn1
        bkn1 = self._norm_bkin(input_logical_ch_bkn1)
        bkn2 = None if input_logical_ch_bkn2 is None else self._norm_bkin(input_logical_ch_bkn2)

        if input_logical_ch_bbk.channel != ChannelName.AACH_CHANNEL:
            raise ValueError(f"Passed broadcast block is invalid for {type(self).__name__} expected"
                             f" {ChannelName.AACH_CHANNEL}")
        burst_type, multiple_logical_ch_state, tsi = self._validate_normal_downlink_mapping(bkn1, bkn2)
        self.burst_type = burst_type
        self.mixed_burst = multiple_logical_ch_state
        training_seq_index = tsi
        # Build the burst
        d = self.start_guard_bit_period
        burst_bit_seq = empty(shape=(self.sn_max*2)+d+self.end_guard_bit_period, dtype=uint8)

        burst_bit_seq[d:2+d] = NORMAL_TRAINING_SEQUENCE[2][20:22]
        # temporarily skip phase adjustment bits g - [d+2:4+d]
        burst_bit_seq[d+4:220+d] = input_logical_ch_bkn1.type_5_blocks[0][:216]
        burst_bit_seq[d+220:234+d] = input_logical_ch_bbk.type_5_blocks[0][:14]
        burst_bit_seq[d+234:256+d] = NORMAL_TRAINING_SEQUENCE[training_seq_index][:22]
        burst_bit_seq[d+256:272+d] = input_logical_ch_bbk.type_5_blocks[0][14:30]
        if input_logical_ch_bkn2 is not None and self.mixed_burst:
            burst_bit_seq[d+272:488+d] = input_logical_ch_bkn2.type_5_blocks[0][:216]
        else:
            burst_bit_seq[d+272:488+d] = input_logical_ch_bkn1.type_5_blocks[0][216:432]
        # temporarily skip phase adjustment bits h - [d+488:490+d]
        burst_bit_seq[d+490:492+d] = NORMAL_TRAINING_SEQUENCE[2][:2]
        # Now insert phase adjustment bits
        burst_bit_seq[d+2:4+d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                 PHASE_ADJUSTMENT_SYMBOL_RANGE['g'], d)
        burst_bit_seq[d+488:490+d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                     PHASE_ADJUSTMENT_SYMBOL_RANGE['h'], d)
        # must add guard period training sequence if there is no ramping at the start
        if ramp_up_down_state[0]:
            burst_bit_seq[:d] = zeros(shape=self.start_guard_bit_period, dtype=uint8)
            self.start_ramp_period = d
        else:
            # add preceding bits per 9.4.5.2
            burst_bit_seq[:d] = NORMAL_TRAINING_SEQUENCE[2][10:20]
            self.start_ramp_period = 0
        # must add guard period training sequence if there is no ramping at the end
        if ramp_up_down_state[1]:
            burst_bit_seq[492+d:510] = zeros(shape=self.end_guard_bit_period, dtype=uint8)
            self.end_ramp_period = TIMESLOT_BIT_LENGTH - 492+d
        else:
            # add following bits per 9.4.5.2
            burst_bit_seq[492+d:510] = NORMAL_TRAINING_SEQUENCE[2][2:10]
            self.end_ramp_period = 0
        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:
        raise NotImplementedError

###################################################################################################


class SynchronousDownlinkMixin:

    def _validate_normal_downlink_mapping(self: DownlinkHost, bkn1: str,
                                          bkn2: str | None) -> tuple[BurstContent, bool]:
        """
        Validates (bkn1, bkn2) for synchronous downlink burst usage across CP/TP/UP
        and returns:
            (burst_type: BurstContent, mixedBurst: bool, training_seq_index: int)
        """

        if bkn1 not in (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL):
            raise ValueError(f"For {type(self).__name__}, invalid bkn1:{bkn1} for synchronous downlink")
        if bkn2 not in (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL, ChannelName.BLCH_CHANNEL):
            raise ValueError(f"For {type(self).__name__}, invalid bkn2:{bkn2} for synchronous downlink")
        burst_type = BurstContent.BURST_CONTROL_TYPE
        mixed = True

        match (bkn1, bkn2):
            case (ChannelName.BSCH_CHANNEL, ChannelName.SCH_HD_CHANNEL):
                # BSCH in BKN1, SCH/HD (or BLCH replacing SCH/HD) in BKN2
                # Valid on TP or CP, with control-frame timing and (MN+TN)%4==3
                if self.phy_channel in (PhyType.TRAFFIC_CHANNEL, PhyType.CONTROL_CHANNEL):
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for phy {self.phy_channel} "
                            f"bkn1:{bkn1} bkn2:{bkn2} (expected control frame)"
                        )
                    if ((self.multiframe_number + self.timeslot) % 4) != 3:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 3 for bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiframe_number}, TN={self.timeslot})"
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)
            case (ChannelName.BSCH_CHANNEL, ChannelName.BLCH_CHANNEL):
                # BSCH in BKN1, SCH/HD (or BLCH replacing SCH/HD) in BKN2
                # Valid on TP or CP, with control-frame timing and (MN+TN)%4==3
                if self.phy_channel in (PhyType.TRAFFIC_CHANNEL, PhyType.CONTROL_CHANNEL):
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for phy {self.phy_channel} "
                            f"bkn1:{bkn1} bkn2:{bkn2} (expected control frame)"
                        )
                    if ((self.multiframe_number + self.timeslot) % 4) != 3:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 3 for bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiframe_number}, TN={self.timeslot})"
                        )
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.BSCH_CHANNEL, ChannelName.BNCH_CHANNEL):
                # BSCH in BKN1, BNCH in BKN2: only permitted on UP, FN:[1..18]
                if self.phy_channel != PhyType.UNASGN_CHANNEL:
                    raise ValueError(
                        f"For {type(self).__name__}, phy {self.phy_channel} invalid for bkn1:{bkn1} bkn2:{bkn2} "
                        f"(expected UNASGN/UP)"
                    )
                if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                    raise ValueError(f"For {type(self).__name__}, FN {self.frame_number}"
                                     f" invalid for UP bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BNCH_CHANNEL):
                # SCH/HD in BKN1, BNCH in BKN2:
                # - on TP: FN == control frame and (MN+TN)%4==1
                # - on CP: FN:[1..18], and if FN==control frame then (MN+TN)%4==1
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frame_number}"
                                         f" invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                    if ((self.multiframe_number + self.timeslot) % 4) != 1:
                        raise ValueError(
                            f"For {type(self).__name__}, (MN+TN)%4 != 1 for TP bkn1:{bkn1} bkn2:{bkn2} "
                            f"(MN={self.multiframe_number}, TN={self.timeslot})")
                elif self.phy_channel == PhyType.CONTROL_CHANNEL:
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frame_number}"
                                         f" invalid for CP bkn1:{bkn1} bkn2:{bkn2}")
                    if self.frame_number == CONTROL_FRAME_NUMBER:
                        if ((self.multiframe_number + self.timeslot) % 4) != 1:
                            raise ValueError(
                                f"For {type(self).__name__}, (MN+TN)%4 != 1 for CP control frame "
                                f"(MN={self.multiframe_number}, TN={self.timeslot})")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)

            case (ChannelName.SCH_HD_CHANNEL, ChannelName.BLCH_CHANNEL):
                # SCH/HD in BKN1, and BKN2 is SCH/HD (or BLCH replacing it):
                # - on TP: FN == control frame
                # - on CP or UP: FN:[1..18]
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frame_number}"
                                         f" invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                elif self.phy_channel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number} invalid for phy {self.phy_channel} "
                            f"bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)
            case (ChannelName.SCH_HD_CHANNEL, _):
                if bkn2 != ChannelName.SCH_HD_CHANNEL:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                # SCH/HD in BKN1, and BKN2 is SCH/HD (or BLCH replacing it):
                # - on TP: FN == control frame
                # - on CP or UP: FN:[1..18]
                if self.phy_channel == PhyType.TRAFFIC_CHANNEL:
                    if self.frame_number != CONTROL_FRAME_NUMBER:
                        raise ValueError(f"For {type(self).__name__}, FN {self.frame_number}"
                                         f" invalid for TP bkn1:{bkn1} bkn2:{bkn2}")
                elif self.phy_channel in (PhyType.CONTROL_CHANNEL, PhyType.UNASGN_CHANNEL):
                    if not 1 <= self.frame_number <= MULTIFRAME_TDMAFRAME_LENGTH:
                        raise ValueError(
                            f"For {type(self).__name__}, FN {self.frame_number}"
                            f"invalid for phy {self.phy_channel} bkn1:{bkn1} bkn2:{bkn2}")
                else:
                    raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                     f" invalid for bkn1:{bkn1} bkn2:{bkn2}")
                return (burst_type, mixed)
            case _:
                raise ValueError(f"For {type(self).__name__}, invalid combination bkn1:{bkn1} bkn2:{bkn2}")

###################################################################################################


class SyncContDownlinkBurst(SynchronousDownlinkMixin, DownlinkHost, Burst):
    sn_max = 255
    start_guard_bit_period = 0
    end_guard_bit_period = 0
    subslot_width = 2
    link_direction = LinkDirection.DOWNLINK
    burst_type = BurstContent.BURST_CONTROL_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    phy_channel: PhyType
    frame_number: int
    multiframe_number: int
    timeslot: int

    def construct_burst_sequence(self, input_logical_ch_sb: BSCH | SCH_HD,
                                 input_logical_ch_bbk: AACH,
                                 input_logical_ch_bkn2: BNCH | BLCH | SCH_HD,
                                 ramp_up_down_state: tuple[bool, bool] = (False, False)) -> NDArray[uint8]:
        if input_logical_ch_bbk.channel != ChannelName.AACH_CHANNEL:
            raise ValueError(f"Passed broadcast block is invalid for {type(self).__name__} expected"
                             f" {ChannelName.AACH_CHANNEL}")

        bkn1 = input_logical_ch_sb.channel
        bkn2 = input_logical_ch_bkn2.channel
        _, multiple_logical_ch_state = self._validate_normal_downlink_mapping(bkn1, bkn2)
        self.mixed_burst = multiple_logical_ch_state

        # Build the burst
        burst_bit_seq = empty(shape=(self.sn_max*2), dtype=uint8)

        if ramp_up_down_state[0]:
            # if we are ramp up (TRUE), it means that this is the first burst
            burst_bit_seq[:12] = zeros(shape=self.start_guard_bit_period, dtype=uint8)
            self.start_ramp_period = 12
        else:
            # other we are continuous (or we are ramping down add preceding bits per 9.4.5.1 - Table 28)
            burst_bit_seq[:12] = NORMAL_TRAINING_SEQUENCE[2][10:22]
            self.start_ramp_period = 0

        # temporarily skip phase adjustment bits C - [12:14]
        burst_bit_seq[14:94] = FREQUENCY_CORRECTION_FIELD
        burst_bit_seq[94:214] = input_logical_ch_sb.type_5_blocks[0][:120]
        burst_bit_seq[214:252] = SYNCHRONIZATION_TRAINING_SEQUENCE
        burst_bit_seq[252:282] = input_logical_ch_bbk.type_5_blocks[0]
        burst_bit_seq[282:498] = input_logical_ch_bkn2.type_5_blocks[0][:216]   # type: ignore[attr-defined]
        # temporarily skip phase adjustment bits D - [498:500]
        if ramp_up_down_state[1]:
            # if we are ramp down (TRUE), it means that this is the last burst we are ramping down
            burst_bit_seq[500:510] = zeros(shape=self.end_guard_bit_period, dtype=uint8)
            self.end_ramp_period = 10
        else:
            # otherwise we are continuous (or we are have ramped up add preceding bits per 9.4.5.1 - Table 27)
            burst_bit_seq[500:510] = NORMAL_TRAINING_SEQUENCE[2][0:10]
            self.end_ramp_period = 0

        # Now insert phase adjustment bits
        burst_bit_seq[12:14] = calculate_phase_adjustment_bits(burst_bit_seq, PHASE_ADJUSTMENT_SYMBOL_RANGE['c'], 0)
        burst_bit_seq[498:500] = calculate_phase_adjustment_bits(burst_bit_seq, PHASE_ADJUSTMENT_SYMBOL_RANGE['d'], 0)
        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:
        raise NotImplementedError

###################################################################################################


class SyncDiscontDownlinkBurst(SynchronousDownlinkMixin, DownlinkHost, Burst):
    sn_max = 246
    start_guard_bit_period = 10
    end_guard_bit_period = 8
    subslot_width = 2
    link_direction = LinkDirection.DOWNLINK
    burst_type = BurstContent.BURST_CONTROL_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    phy_channel: PhyType
    frame_number: int
    multiframe_number: int
    timeslot: int

    def construct_burst_sequence(self, input_logical_ch_sb: BSCH | SCH_HD,
                                 input_logical_ch_bbk: AACH,
                                 input_logical_ch_bkn2: BNCH | BLCH | SCH_HD,
                                 ramp_up_down_state: tuple[bool, bool] = (True, True)) -> NDArray[uint8]:
        if input_logical_ch_bbk.channel != ChannelName.AACH_CHANNEL:
            raise ValueError(f"Passed broadcast block is invalid for {type(self).__name__} expected"
                             f" {ChannelName.AACH_CHANNEL}")

        bkn1 = input_logical_ch_sb.channel
        bkn2 = input_logical_ch_bkn2.channel

        _, multiple_logical_ch_state = self._validate_normal_downlink_mapping(bkn1, bkn2)
        self.mixed_burst = multiple_logical_ch_state

        # Build the burst
        d = self.start_guard_bit_period
        burst_bit_seq = empty(shape=(self.sn_max*2)+d+self.end_guard_bit_period, dtype=uint8)

        burst_bit_seq[d:2+d] = NORMAL_TRAINING_SEQUENCE[2][20:22]
        # temporarily skip phase adjustment bits i - [d+2:4+d]
        burst_bit_seq[d+4:84+d] = FREQUENCY_CORRECTION_FIELD
        burst_bit_seq[d+84:204+d] = input_logical_ch_sb.type_5_blocks[0][:120]
        burst_bit_seq[d+204:242+d] = SYNCHRONIZATION_TRAINING_SEQUENCE
        burst_bit_seq[d+242:272+d] = input_logical_ch_bbk.type_5_blocks[0][:30]
        burst_bit_seq[d+272:488+d] = input_logical_ch_bkn2.type_5_blocks[0][:216]
        # temporarily skip phase adjustment bits j - [d+488:490+d]
        burst_bit_seq[d+490:492+d] = NORMAL_TRAINING_SEQUENCE[2][:2]

        # Now insert phase adjustment bits
        burst_bit_seq[d+2:4+d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                 PHASE_ADJUSTMENT_SYMBOL_RANGE['i'], d)
        burst_bit_seq[d+488:490+d] = calculate_phase_adjustment_bits(burst_bit_seq,
                                                                     PHASE_ADJUSTMENT_SYMBOL_RANGE['j'], d)
        # must add guard period training sequence if there is no ramping at the start
        if ramp_up_down_state[0]:
            burst_bit_seq[:d] = zeros(shape=self.start_guard_bit_period, dtype=uint8)
            self.start_ramp_period = d
        else:
            # add preceding bits per 9.4.5.2
            burst_bit_seq[:d] = NORMAL_TRAINING_SEQUENCE[2][10:20]
            self.start_ramp_period = 0
        # must add guard period training sequence if there is no ramping at the end
        if ramp_up_down_state[1]:
            burst_bit_seq[492+d:510] = zeros(shape=self.end_guard_bit_period, dtype=uint8)
            self.end_ramp_period = TIMESLOT_BIT_LENGTH - (492+d)
        else:
            # add following bits per 9.4.5.2
            burst_bit_seq[492+d:510] = NORMAL_TRAINING_SEQUENCE[2][2:10]
            self.end_ramp_period = 0
        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:
        raise NotImplementedError

###################################################################################################


class LinearizationUplinkBurst(Burst):
    sn_max = 240
    start_guard_bit_period = 34    # not in the standard just chosen
    end_guard_bit_period = 15
    subslot_width = 1
    link_direction = LinkDirection.UPLINK
    burst_type = BurstContent.BURST_LINEARIZATION_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def construct_burst_sequence(self, input_logical_ch_ssn1: CLCH) -> NDArray[uint8]:
        if self.phy_channel in (PhyType.TRAFFIC_CHANNEL, PhyType.CONTROL_CHANNEL):
            if self.frame_number != CONTROL_FRAME_NUMBER:
                raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                                 f", FN {self.frame_number} invalid for ssn{self.subslot}"
                                 f": {input_logical_ch_ssn1.channel}")
            if (self.multiframe_number + self.timeslot) % 4 != 3:
                raise ValueError(f"For {type(self).__name__}, (MN+TN)%4 != 3 for CP control frame "
                                 f"(MN={self.multiframe_number}, TN={self.timeslot})")  # per 9.5.2 (74)
        if self.subslot != 1:
            raise ValueError(f" For {type(self).__name__}, subslot {self.subslot} is invalid, expected (1)")

        if input_logical_ch_ssn1.channel != ChannelName.CLCH_CHANNEL:
            raise ValueError(f"For {type(self).__name__}, phy {self.phy_channel}"
                             f", invalid ssn of {input_logical_ch_ssn1.channel}")

        # Build the burst
        d = self.start_guard_bit_period
        burst_bit_seq = empty(shape=(self.sn_max*2)+d+self.end_guard_bit_period, dtype=uint8)
        burst_bit_seq[0:d] = zeros(shape=self.start_guard_bit_period, dtype=uint8)
        burst_bit_seq[d:206+d] = input_logical_ch_ssn1.type_5_blocks[0][:206]
        # End guard bits
        burst_bit_seq[d+206:255] = zeros(shape=self.end_guard_bit_period, dtype=uint8)
        self.start_ramp_period = self.start_guard_bit_period
        self.end_ramp_period = self.end_guard_bit_period
        return burst_bit_seq

    def deconstuct_burst_sequence(self) -> None:
        raise NotImplementedError

###################################################################################################


class NullHalfslotUplinkBurst(Burst):
    sn_max = 255
    start_guard_bit_period = 0
    end_guard_bit_period = 0
    subslot_width = 1
    link_direction = LinkDirection.UPLINK
    burst_type = BurstContent.BURST_MIXED_TYPE

    ALLOWED_PHY = {PhyType.CONTROL_CHANNEL, PhyType.TRAFFIC_CHANNEL, PhyType.UNASGN_CHANNEL}

    def construct_burst_sequence(self) -> NDArray[uint8]:
        return zeros(shape=self.sn_max, dtype=uint8)

    def deconstuct_burst_sequence(self) -> NDArray[uint8]:
        return zeros(shape=self.sn_max, dtype=uint8)

###################################################################################################
