"""
constants.py holds constains that are used across modules, including enum string types that define logical and physical
channel behaviours as well as MAC layer operation types. For modules specific constants they are included within modules
as needed or otherwise colated into related utility files.
"""
from dataclasses import dataclass
from enum import Enum
#############################################
# PHYSICAL CHANNEL CONSTANTS


class PhyType(str, Enum):
    """
    Enum class used to wrap the valid types of physical channel for a physical channel object
    """
    TRAFFIC_CHANNEL = "TP"
    CONTROL_CHANNEL = "CP"
    UNASGN_CHANNEL = "UP"


class LinkDirection(str, Enum):
    """
    Enum class used to wrap the valid link directions for a physical channels object
    """
    UPLINK = "UL"
    DOWNLINK = "DL"

#############################################
# BURST CONSTANTS


class BurstContent(str, Enum):
    """
    Enum class used to wrap a descriptor for the content of a burst, used for a physical layer burst object
    """
    BURST_TRAFFIC_TYPE = "traffic"
    BURST_CONTROL_TYPE = "control"
    BURST_MIXED_TYPE = "mixed"
    BURST_LINEARIZATION_TYPE = "linear"


class SlotLength(str, Enum):
    """
    Enum class used to wrap a the a descriptor of wether or not a physical layer burst object uses an entire slot
    of half, such as in the case of uplink control and linearization bursts.
    """
    HALF_SUBSLOT = "half"
    FULL_SUBSLOT = "full"

#############################################
# LOGICAL CHANNEL CONSTANTS


class ChannelKind(str, Enum):
    """
    Enum class used to wrap the valid types of logical channel for a logical channel object
    """
    TRAFFIC_TYPE = "traffic"
    CONTROL_TYPE = "control"
    LINEARIZATION_TYPE = "linear"


# Channel variable names for MAC burst building
class ChannelName(str, Enum):
    """
    Enum class used to wrap the valid names of logical channels for a logical channel object used in verifying burst
    building to prevent invalid combinations
    """
    BNCH_CHANNEL = "BNCH"
    BSCH_CHANNEL = "BSCH"
    SCH_CHANNEL = "SCH"
    SCH_F_CHANNEL = "SCH/F"
    SCH_HD_CHANNEL = "SCH/HD"
    SCH_HU_CHANNEL = "SCH/HU"
    AACH_CHANNEL = "AACH"
    STCH_CHANNEL = "STCH"
    TCH_CHANNEL = "TCH"
    TCH_S_CHANNEL = "TCH/S"
    TCH_7_2_CHANNEL = "TCH/7.2"
    TCH_4_8_CHANNEL = "TCH/4.8"
    TCH_2_4_CHANNEL = "TCH/2.4"
    CLCH_CHANNEL = "CLCH"
    BLCH_CHANNEL = "BLCH"


SUBSLOT_BIT_LENGTH = 255        # The number of modulation bits in a subslot
TIMESLOT_SYMBOL_LENGTH = 255    # The number of symbols in a full slot

#############################################
# MAC LAYER CONSTANTS


# Control Modes
class ControlMode(str, Enum):
    """
    Enum class used to wrap the valid types of control modes for BS and MS MAC layer control plane signalling
    """
    NCM_CTRL_MODE = "NCM"       # Normal Control Mode
    MC_CTRL_MODE = "MC"         # Minimum Control Mode


# Transmission Modes
class TransmissionMode(str, Enum):
    """
    Enum class used to wrap the valid types of transmission modes for BS and MS MAC layer traffic plane signalling
    """
    D_CT_BS_MODE = "D-CT"        # Downlink-Continuous Transmission (D-CT)
    D_CTT_BS_MODE = "D-CTT"      # Downlink-Carrier Timesharing Transmission (D-CTT)
    D_MCCTT_BS_MODE = "D-MCCTT"  # Downlink-Main Control Channel Timesharing Transmission (D-MCCTT)
    U_MST_BS_MODE = "U-MST"      # Multiple Slot Transmission

#############################################
# TRANSCIEVER CONSTANTS


TETRA_SYMBOL_RATE = 18_000  # The base EN 300 392-2 symbol rate

TX_BB_SAMPLING_FACTOR = 64  # The culmative oversampling factor of the tx baseband processing

TRANSMIT_SIMULATION_SAMPLING_FACTOR = 10  # The internal sw simulator sampling factor over the DAC rate

# Following constants Per EN 300 392-2 V2.4.2 - 9.3
HYPERFRAME_MULTIFRAME_LENGTH = 60   # Number of multiframes in a hyperframe

MULTIFRAME_TDMAFRAME_LENGTH = 18    # Number of frames in a multiframe

CONTROL_FRAME_NUMBER = 18           # The frame number (FN) of the control frame (last frame in multiframe)

TDMAFRAME_TIMESLOT_LENGTH = 4       # How many timeslots in a frame

TIMESLOT_BIT_LENGTH = 510           # How many modulation bits in a timeslot

TIMESLOT_SUBSLOT_LENGTH = 2         # How many subslots in a timeslot

OPENTETRAPHYMAC_DEFAULT_TX_FREQUENCY = 905.025E6
OPENTETRAPHYMAC_DEFAULT_RX_FREQUENCY = 918.025E6

#############################################
# Propagation Models


class TetraTapGainProcess(str, Enum):
    STATIC_PROCESS = "STATIC"
    RICE_PROCESS = "RICE"
    CLASS_PROCESS = "CLASS"


class TetraPropagationModels(str, Enum):
    STATIC = "STATIC"
    RURAL_AREA = "RA"
    TYPICAL_URBAN = "TU"
    BAD_URBAN = "BU"
    HILLY_TERRAIN = "HT"
    EQUALIZATION_TEST = "EQ"


# Velocities (speeds actually) are only defined for STATIC, TU50, HT200, EQ200, other models are included here but don't
# have a default velocity defined in the standard because they arent used for performance requirements
TETRA_DEFAULT_MODEL_VELOCITIES_KPH = {TetraPropagationModels.STATIC: 0.0,
                                      TetraPropagationModels.TYPICAL_URBAN: 50.0,
                                      TetraPropagationModels.HILLY_TERRAIN: 200.0,
                                      TetraPropagationModels.EQUALIZATION_TEST: 200.0}


@dataclass(frozen=True)
class PropagationTapParameters:
    delay: float
    amplitude_scale: float
    process: TetraTapGainProcess


@dataclass(frozen=True)
class PropagationModelParameters:
    taps: tuple[PropagationTapParameters, ...]


TETRA_PROPAGATION_MODELS = {
    TetraPropagationModels.STATIC: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.STATIC_PROCESS),)),

    TetraPropagationModels.RURAL_AREA: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.RICE_PROCESS),)),

    TetraPropagationModels.TYPICAL_URBAN: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(5.0E-6, 10**(-22.3/20), TetraTapGainProcess.CLASS_PROCESS))),

    TetraPropagationModels.BAD_URBAN: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(5.0E-6, 10**(-3.0/20), TetraTapGainProcess.CLASS_PROCESS))),

    TetraPropagationModels.HILLY_TERRAIN: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(15.0E-6, 10**(-8.6/20), TetraTapGainProcess.CLASS_PROCESS))),

    TetraPropagationModels.EQUALIZATION_TEST: PropagationModelParameters(
        taps=(PropagationTapParameters(0.0, 1.0, TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(11.6E-6, 1.0, TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(73.2E-6, 10**(-10.2/20), TetraTapGainProcess.CLASS_PROCESS),
              PropagationTapParameters(99.3E-6, 10**(-16.0/20), TetraTapGainProcess.CLASS_PROCESS)))}


TETRA_FADING_SIMULATION_RATE = int(80_000)


# Enum constant class used to specify the nature of the burst/block being handled
# by PropagationTap
class StreamPosition(str, Enum):
    ISOLATED_BURST = "ISOLATED"
    START_BURST = "START"
    MIDDLE_BURST = "MIDDLE"
    END_BURST = "END"


#############################################
