"""
constants.py holds constains that are used across modules, including enum string types that define logical and physical
channel behaviours as well as MAC layer operation types. For modules specific constants they are included within modules
as needed or otherwise colated into related utility files.
"""
from dataclasses import dataclass
from enum import Enum

from numpy import array, float64, int64
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


class StationType(str, Enum):
    """
    Enum class used to wrap the valid station types
    """
    MOBILE_STATION = "MS"
    BASE_STATION = "BS"

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
    BURST_UNKNOWN_TYPE = "unknown"


class SlotLength(str, Enum):
    """
    Enum class used to wrap a the a descriptor of wether or not a physical layer burst object uses an entire slot
    of half, such as in the case of uplink control and linearization bursts.
    """
    HALF_SUBSLOT = "half"
    FULL_SUBSLOT = "full"


class BurstContinuity(str, Enum):
    """
    Enum class used to wrap how a given burst type is with respect to its' continuity with subsequent/preceeding bursts
    """
    ISOLATED = "isolated"   # Burst type is always isolated (ramps up/down)
    REQUIRED = "required"   # Burst type is required to be continuous where possible
    # Burst type can be either be isolated or continuous depending on mode of operation
    # this applies only to NormalDownlinkBurst
    OPTIONAL = "optional"

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

# Internal sw simulator sampling rate, allows for capture of harmonics
TETRA_TX_SIMULATION_SAMPLE_RATE = int(TX_BB_SAMPLING_FACTOR * TETRA_SYMBOL_RATE * TRANSMIT_SIMULATION_SAMPLING_FACTOR)


#############################################
# Propagation Models


class TetraTapGainProcess(str, Enum):
    """
    Enum class used to wrap the types of fading gain procceses that the TETRA standard specifies
    """
    STATIC_PROCESS = "STATIC"  # Static process only has doppler shift, constant envelope
    RICE_PROCESS = "RICE"      # Rice process is equal combination of LOS static and class(rayleigh) processes
    CLASS_PROCESS = "CLASS"    # Class process is Rayleigh fading process


class TetraPropagationModels(str, Enum):
    """
    Enum class used to wrap the propagation model types that the TETRA standard documents
    """
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
    """
    Dataclass used to hold PropagationTap data such as process type, delay, and mean scaling
    """
    delay: float
    amplitude_scale: float
    process: TetraTapGainProcess


@dataclass(frozen=True)
class PropagationModelParameters:
    """
    Dataclass used to hold up to multiple PropagationTapParameters describing the various taps of TETRA propgation
    model
    """
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
    """
    Enum class used to wrap the valid descriptions of block characteristics used in PropgationTap/Mode implementation
    """
    ISOLATED_BURST = "ISOLATED"
    START_BURST = "START"
    MIDDLE_BURST = "MIDDLE"
    END_BURST = "END"


#############################################
# TETRA Baseband Processing Filter Constants

TX_RRC_Q17_COEFFICIENTS = array(
    [6, 60, 98, 109, 87, 37, -30, -93, -135, -141, -104, -30, 64, 153, 212, 222, 173, 74, -55, -181,
     -269, -292, -237, -115, 47, 204, 308, 319, 221, 26, -221, -452, -589, -564, -342, 66, 592, 1122,
     1511, 1613, 1321, 594, -510, -1834, -3125, -4075, -4361, -3712, -1958, 923, 4783, 9308, 14052,
     18499, 22130, 24505, 25332, 24505, 22130, 18499, 14052, 9308, 4783, 923, -1958, -3712, -4361,
     -4075, -3125, -1834, -510, 594, 1321, 1613, 1511, 1122, 592, 66, -342, -564, -589, -452, -221,
     26, 221, 319, 308, 204, 47, -115, -237, -292, -269, -181, -55, 74, 173, 222, 212, 153, 64, -30,
     -104, -141, -135, -93, -30, 37, 87, 109, 98, 60, 6], dtype=int64)

TX_RRC_FLOAT_COEFFICIENTS = array(
    [4.6028807E-05, 4.5548688E-04, 7.4681803E-04, 8.2835555E-04, 6.6207664E-04, 2.7856705E-04,
     -2.2615044E-04, -7.1247749E-04, -1.0332032E-03, -1.0747046E-03, -7.9234410E-04, -2.2970411E-04,
     4.8512162E-04, 1.1672165E-03, 1.6210295E-03, 1.6939947E-03, 1.3234488E-03, 5.6322449E-04,
     -4.1943332E-04, -1.3787722E-03, -2.0509101E-03, -2.2257303E-03, -1.8117221E-03, -8.7512349E-04,
     3.6038237E-04, 1.5580310E-03, 2.3496656E-03, 2.4371645E-03, 1.6885466E-03, 2.0193664E-04,
     -1.6844368E-03, -3.4467725E-03, -4.4899732E-03, -4.3008132E-03, -2.6063069E-03, 5.0252874E-04,
     4.5184297E-03, 8.5619008E-03, 1.1526610E-02, 1.2308078E-02, 1.0075723E-02, 4.5330846E-03,
     -3.8935654E-03, -1.3989874E-02, -2.3842659E-02, -3.1086056E-02, -3.3274468E-02, -2.8323304E-02,
     -1.4939181E-02, 7.0435614E-03, 3.6491636E-02, 7.1012035E-02, 1.0720996E-01, 1.4113323E-01,
     1.6883495E-01, 1.8696181E-01, 1.9326745E-01, 1.8696181E-01, 1.6883495E-01, 1.4113323E-01,
     1.0720996E-01, 7.1012035E-02, 3.6491636E-02, 7.0435614E-03, -1.4939181E-02, -2.8323304E-02,
     -3.3274468E-02, -3.1086056E-02, -2.3842659E-02, -1.3989874E-02, -3.8935654E-03, 4.5330846E-03,
     1.0075723E-02, 1.2308078E-02, 1.1526610E-02, 8.5619008E-03, 4.5184297E-03, 5.0252874E-04,
     -2.6063069E-03, -4.3008132E-03, -4.4899732E-03, -3.4467725E-03, -1.6844368E-03, 2.0193664E-04,
     1.6885466E-03, 2.4371645E-03, 2.3496656E-03, 1.5580310E-03, 3.6038237E-04, -8.7512349E-04,
     -1.8117221E-03, -2.2257303E-03, -2.0509101E-03, -1.3787722E-03, -4.1943332E-04, 5.6322449E-04,
     1.3234488E-03, 1.6939947E-03, 1.6210295E-03, 1.1672165E-03, 4.8512162E-04, -2.2970411E-04,
     -7.9234410E-04, -1.0747046E-03, -1.0332032E-03, -7.1247749E-04, -2.2615044E-04, 2.7856705E-04,
     6.6207664E-04, 8.2835555E-04, 7.4681803E-04, 4.5548688E-04, 4.6028807E-05], dtype=float64)

TX_LPF_Q17_COEFFICIENTS = array(
    [40, 136, 205, 218, 156, 27, -136, -283, -361, -329, -179, 57, 312, 503, 555, 429, 140,
     -240, -595, -804, -776, -485, 9, 570, 1022, 1199, 1004, 447, -339, -1128, -1659, -1720,
     -1216, -226, 1003, 2104, 2694, 2483, 1388, -411, -2476, -4191, -4904, -4088, -1501, 2723,
     8064, 13701, 18671, 22078, 23290, 22078, 18671, 13701, 8064, 2723, -1501, -4088, -4904,
     -4191, -2476, -411, 1388, 2483, 2694, 2104, 1003, -226, -1216, -1720, -1659, -1128, -339,
     447, 1004, 1199, 1022, 570, 9, -485, -776, -804, -595, -240, 140, 429, 555, 503, 312, 57,
     -179, -329, -361, -283, -136, 27, 156, 218, 205, 136, 40], dtype=int64)

TX_LPF_FLOAT_COEFFICIENTS = array(
    [3.0866607E-04, 1.0378698E-03, 1.5670853E-03, 1.6598026E-03, 1.1896548E-03, 2.0867007E-04,
     -1.0375803E-03, -2.1627121E-03, -2.7524120E-03, -2.5066389E-03, -1.3648995E-03, 4.3268623E-04,
     2.3801184E-03, 3.8384833E-03, 4.2354544E-03, 3.2718693E-03, 1.0650634E-03, -1.8293252E-03,
     4.5416570E-03, -6.1367146E-03, -5.9191459E-03, -3.7006680E-03, 6.5840668E-05, 4.3479592E-03,
     7.7964842E-03, 9.1487500E-03, 7.6578711E-03, 3.4103186E-03, -2.5865371E-03, -8.6053726E-03,
     -1.2660975E-02, -1.3118963E-02, -9.2759097E-03, -1.7231343E-03, 7.6487811E-03, 1.6053667E-02,
     2.0551295E-02, 1.8945906E-02, 1.0593299E-02, -3.1324285E-03, -1.8888838E-02, -3.1977766E-02,
     -3.7411781E-02, -3.1189190E-02, -1.1454434E-02, 2.0773036E-02, 6.1524669E-02, 1.0452736E-01,
     1.4244613E-01, 1.6844460E-01, 1.7769138E-01, 1.6844460E-01, 1.4244613E-01, 1.0452736E-01,
     6.1524669E-02, 2.0773036E-02, -1.1454434E-02, -3.1189190E-02, -3.7411781E-02, -3.1977766E-02,
     -1.8888838E-02, -3.1324285E-03, 1.0593299E-02, 1.8945906E-02, 2.0551295E-02, 1.6053667E-02,
     7.6487811E-03, -1.7231343E-03, -9.2759097E-03, -1.3118963E-02, -1.2660975E-02, -8.6053726E-03,
     -2.5865371E-03, 3.4103186E-03, 7.6578711E-03, 9.1487500E-03, 7.7964842E-03, 4.3479592E-03,
     6.5840668E-05, -3.7006680E-03, -5.9191459E-03, -6.1367146E-03, -4.5416570E-03, -1.8293252E-03,
     1.0650634E-03, 3.2718693E-03, 4.2354544E-03, 3.8384833E-03, 2.3801184E-03, 4.3268623E-04,
     -1.3648995E-03, -2.5066389E-03, -2.7524120E-03, -2.1627121E-03, -1.0375803E-03, 2.0867007E-04,
     1.1896548E-03, 1.6598026E-03, 1.5670853E-03, 1.0378698E-03, 3.0866607E-04], dtype=float64)

TX_HALFBAND1_Q17_COEFFICIENTS = array(
    [-145, 0, 193, 0, -323, 0, 555, 0, -914, 0, 1434, 0, -2170, 0, 3221, 0, -4805, 0,
     7491, 0, -13392, 0, 41587, 65606, 41587, 0, -13392, 0, 7491, 0, -4805, 0, 3221, 0,
     -2170, 0, 1434, 0, -914, 0, 555, 0, -323, 0, 193, 0, -145], dtype=int64)

TX_HALFBAND1_FLOAT_COEFFICIENTS = array(
    [-1.1083445E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -2.4647851E-03, 0.0000000E+00,
     4.2366369E-03, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 1.0942169E-02, 0.0000000E+00,
     -1.6552123E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -3.6657065E-02, 0.0000000E+00,
     5.7154625E-02, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 3.1728380E-01, 5.0053275E-01,
     3.1728380E-01, 0.0000000E+00, -1.0217133E-01, 0.0000000E+00, 5.7154625E-02, 0.0000000E+00,
     -3.6657065E-02, 0.0000000E+00, 2.4572962E-02, 0.0000000E+00, -1.6552123E-02, 0.0000000E+00,
     1.0942169E-02, 0.0000000E+00, -6.9756542E-03, 0.0000000E+00, 4.2366369E-03, 0.0000000E+00,
     -2.4647851E-03, 0.0000000E+00, 1.4727359E-03, 0.0000000E+00, -1.1083445E-03], dtype=float64)

TX_HALFBAND2_Q17_COEFFICIENTS = array(
    [-304, 0, 711, 0, -2084, 0, 5063, 0, -11725, 0, 41035, 65681, 41035, 0, -11725, 0,
     5063, 0, -2084, 0, 711, 0, -304], dtype=int64)

TX_HALFBAND2_FLOAT_COEFFICIENTS = array(
    [-2.3200984E-03, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -1.5900960E-02, 0.0000000E+00,
     3.8630295E-02, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.1306928E-01, 5.0110528E-01,
     3.1306928E-01, 0.0000000E+00, -8.9455216E-02, 0.0000000E+00, 3.8630295E-02, 0.0000000E+00,
     -1.5900960E-02, 0.0000000E+00, 5.4240586E-03, 0.0000000E+00, -2.3200984E-03], dtype=float64)

TX_HALFBAND3_Q17_COEFFICIENTS = array(
    [0, 1181, 0, -7504, 0, 39118, 65482, 39118, 0, -7504, 0, 1181, 0], dtype=int64)

TX_HALFBAND3_FLOAT_COEFFICIENTS = array(
    [0.0000000E+00, 9.0088874E-03, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 2.9844614E-01,
     4.9958680E-01, 2.9844614E-01, 0.0000000E+00, -5.7248430E-02, 0.0000000E+00, 9.0088874E-03,
     0.0000000E+00], dtype=float64)

#############################################
