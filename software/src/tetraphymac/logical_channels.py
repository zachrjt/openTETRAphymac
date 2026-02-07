"""
logical_channels.py contains the logical channels as described in EN 300 392-2 V2.4.2. Logical channels are the basis
for bursts and exist at the upper PHY layer, their purpose is to take in control plane or traffic plane data from the
MAC and perform the appropriate CRC, interleaving, encoding, and interleaving on data before the resultant type 5 blocks
are used in phyiscal channel bursts as uplink/downlink control, traffic, broadcast, or linearization blocks/subslots
"""
from typing import Literal
from numpy.random import randint
from numpy import uint8, array, empty
from numpy.typing import NDArray

from .constants import SlotLength, ChannelKind, ChannelName
from .coding_scrambling import crc16_decoder, crc16_encoder, rcpc_decoder, rcpc_encoder, \
    rm3014_decoder, rm3014_encoder, block_deinterleaver, descrambler, n_block_interleaver, block_interleaver, \
    n_block_deinterleaver, scrambler

# Per EN 300 392-2 V2.4.2 - 8.3
LOGICAL_CH_TAIL_BITS = array([0, 0, 0, 0], uint8)  # The zero-valued tail bits appended to type-1 bits


class LogicalChannelVD():
    '''
    logical channels contain:
        1. a method to construct type-5 bits in type 5 blocks
        from type-1 information bits in type-1 blocks from MAC

        2. a method to construct type-1 bits in type 1 blocks from
        type-5 recieved bits in type-5 blocks from PHY
    '''

    # attributes contain the various types of bits, all are stored for now for analysis when needed
    type_5_blocks: NDArray[uint8]
    type_4_blocks: NDArray[uint8]
    type_3_blocks: NDArray[uint8]
    type_2_blocks: NDArray[uint8]
    type_1_blocks: NDArray[uint8]

    channel = ""      # Describes the written name of the logical channel
    channel_type = ""  # Channel type either traffic or control

    m = 0             # Number of input blocks

    k1 = 0            # Number of input bits per block
    k5 = 0            # Number of output bits per block

    crc_result = 0

    def generate_rnd_input(self, m: int = 1) -> NDArray[uint8]:
        """
        Generates m-blocks of self.k1 number of type-1 bits that are random 0's and 1's stored in uint8 numpy array

        :param self: Logical channel child implementation
        :param m: Number of blocks, default is 1
        :type m: int
        :return: k1 (depedant on specific logical ch) numpy random 0's and 1's
         stored in uint8 numpy array
        :rtype: NDArray[uint8]
        """
        return randint(0, 2, size=(m, self.k1), dtype=uint8)

    def validate_k_length(self, k: Literal[1] | Literal[5]):
        """
        Validates the number of bits in the k_ type blocks of the logical channel for k=1 or 5

        :param self: Logical channel child implementation
        :param k: either verifiy type_1_blocks for k=1, or type_5_blocks for k=5
        :type k: Literal[1] | Literal[5]
        """

        if self.m == 0:
            raise ValueError("m unspecified")
        if k == 5:
            if self.k5 == 0:
                raise ValueError("k5 unspecified")
            if self.type_5_blocks.size == 0:
                raise RuntimeError("type_5_blocks unspecified")
            if self.type_5_blocks.shape[1] != self.k5:
                raise RuntimeError(f"For logical channel {self.__class__.__name__}"
                                   f", expected k5: {self.k5} output bits, recieved {self.type_5_blocks.shape}")
        elif k == 1:
            if self.k1 == 0:
                raise ValueError("k1 unspecified")
            if self.type_1_blocks.size == 0:
                raise RuntimeError("type_1_blocks unspecified")
            if self.type_1_blocks.shape[1] != self.k1:
                raise RuntimeError(f"For logical channel {self.__class__.__name__}"
                                   f", expected k1: {self.k1} output bits, recieved {self.type_1_blocks.shape}")

###################################################################################################


# Control CHannels (CCH)
class ControlChannel(LogicalChannelVD):
    '''
    Parent wrapping class for the various child control channel classes. Used to allow for easy grouping of child
    classes in burst building.
    '''

    def __init__(self):
        self.channel_type = ChannelKind.CONTROL_TYPE

###################################################################################################


class BCCH(ControlChannel):
    '''
    Broadcast Control CHannel (BCCH)

    The BCCH shall be a uni-directional channel for common use by all MSs.
    It shall broadcast general information to all MSs
    '''
    def __init__(self):  # pylint: disable=useless-parent-delegation
        super().__init__()


class BNCH(BCCH):
    '''
    Broadcast Network CHannel (BNCH)

    down-link only, broadcasts network information to MSs.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 124
        self.k5 = 216
        self.channel = ChannelName.BNCH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: BNCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 124, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 216
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(self.type_1_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 101)

        self.type_5_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0])
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: BNCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 216, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 124
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 101)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 144, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 124), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)


class BSCH(BCCH):
    '''
    Broadcast Synchronization Channel (BNCH)

    down-link only, broadcast information used for time and scrambling synchronization of the MSs
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 60
        self.k5 = 120
        self.channel = ChannelName.BSCH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: BSCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 60, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 120
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")

        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 80), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(input_data_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 120), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 120), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 11)

        self.type_5_blocks = empty(shape=(self.m, 120), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0], bsch_state=True)
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: BSCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 120, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 60
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)

        self.type_4_blocks = empty(shape=(self.m, 120), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0], bsch_state=True)

        self.type_3_blocks = empty(shape=(self.m, 120), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 11)

        self.type_2_blocks = empty(shape=(self.m, 80), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 80, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 60), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)

###################################################################################################

###################################################################################################


# note linearization channels are not included here since they really only need to be considered at burst level
class SCH(ControlChannel):
    '''
    Signalling CHannel (SCH)

    The SCH shall be shared by all MSs, but may carry messages specific to one or one group of MSs. System operation
    requires the establishment of at least one SCH per BS. SCH may be divided into 3 categories,
    depending on the size of the message:
    '''
    def __init__(self):  # pylint: disable=useless-parent-delegation
        super().__init__()


class SCH_F(SCH):  # pylint: disable=invalid-name
    '''
    Full size Signalling Channel (SCH/F)

    bidirectional channel used for full size messages.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 268
        self.k5 = 432
        self.channel = ChannelName.SCH_F_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: SCH_F logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 268, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 432
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 288), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(self.type_1_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 432), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 432), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 103)

        self.type_5_blocks = empty(shape=(self.m, 432), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0])
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: SCH_F logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 432, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 268
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)

        self.type_4_blocks = empty(shape=(self.m, 432), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_3_blocks = empty(shape=(self.m, 432), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 103)

        self.type_2_blocks = empty(shape=(self.m, 288), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 288, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 268), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)


class SCH_HD(SCH):  # pylint: disable=invalid-name
    '''
    Half size Downlink Signalling Channel (SCH/HD)

    downlink only, used for half size messages.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 124
        self.k5 = 216
        self.channel = ChannelName.SCH_HD_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: SCH_HD logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 124, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 216
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(self.type_1_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 101)

        self.type_5_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0])
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: SCH_HD logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 216, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 124
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 101)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 144, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 124), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)


class SCH_HU(SCH):  # pylint: disable=invalid-name
    '''
    Half size Uplink Signalling Channel (SCH/HU)

    uplink only, used for half size messages.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 92
        self.k5 = 168
        self.channel = ChannelName.SCH_HU_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: SCH_HU logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 92, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 168
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 112), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(input_data_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 168), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 168), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 13)

        self.type_5_blocks = empty(shape=(self.m, 168), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0])
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: SCH_HU logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 168, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 92
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)

        self.type_4_blocks = empty(shape=(self.m, 168), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_3_blocks = empty(shape=(self.m, 168), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 13)

        self.type_2_blocks = empty(shape=(self.m, 112), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 112, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 92), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)


class AACH(ControlChannel):
    '''
    Access Assignment CHannel (AACH)

    The AACH shall be present on all transmitted downlink slots. It shall be used to
    indicate on each physical channel the assignment of the uplink and downlink slots.
    The AACH shall be internal to the MAC.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 14
        self.k5 = 30
        self.channel = ChannelName.AACH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: AACH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 14, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 30
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")

        self.type_2_blocks = empty(shape=(self.m, 30), dtype=uint8)
        self.type_2_blocks[0] = rm3014_encoder(self.type_1_blocks[0])

        self.type_5_blocks = empty(shape=(self.m, 30), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_2_blocks[0], False)
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: AACH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 30, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 16
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_2_blocks = empty(shape=(self.m, 30), dtype=uint8)
        self.type_2_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_1_blocks = empty(shape=(self.m, 14), dtype=uint8)
        self.type_1_blocks[0] = rm3014_decoder(self.type_2_blocks[0])
        self.validate_k_length(1)


class STCH(ControlChannel):
    '''
    STealing CHannel (STCH)

    The STCH is a channel associated to a TCH that temporarily "steals" a part of the associated
    TCH capacity to transmit control messages. It may be used when fast signalling is required.
    In half duplex mode the STCH is unidirectional and has the same direction as the associated TCH.
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 124
        self.k5 = 216
        self.channel = ChannelName.STCH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs encoding of 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks

        :param self: STCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 124, but with only 1 block/row
         to be encoded into self.type_5_blocks of length 216
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0][:-4] = crc16_encoder(self.type_1_blocks[0])
        self.type_2_blocks[0][-4:] = LOGICAL_CH_TAIL_BITS

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = rcpc_encoder(self.type_2_blocks[0], 2, 3)

        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = block_interleaver(self.type_3_blocks[0], 101)

        self.type_5_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_5_blocks[0] = scrambler(self.type_4_blocks[0])
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Performs decoding of 1 block of type-5 bits in input_data_blocks into 1 block within self.type_1_blocks

        :param self: STCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 216, but with only 1 block/row
         to be decoded into self.type_1_blocks of length 124
        :type input_data_blocks: NDArray[uint8]
        """
        self.m = input_data_blocks.shape[0]
        if self.m != 1:
            raise ValueError(f"For type logical channel: {self.__class__.__name__}"
                             f", only 1 data block is supported but {self.m} was passed")
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_4_blocks[0] = descrambler(self.type_5_blocks[0])

        self.type_3_blocks = empty(shape=(self.m, 216), dtype=uint8)
        self.type_3_blocks[0] = block_deinterleaver(self.type_4_blocks[0], 101)

        self.type_2_blocks = empty(shape=(self.m, 144), dtype=uint8)
        self.type_2_blocks[0] = rcpc_decoder(self.type_3_blocks[0], 144, 2, 3)

        self.type_1_blocks = empty(shape=(self.m, 124), dtype=uint8)
        self.type_1_blocks[0], self.crc_result = crc16_decoder(self.type_2_blocks[0][:-4])
        self.validate_k_length(1)

###################################################################################################
# Traffic Channels


class TrafficChannel(LogicalChannelVD):
    '''
    Parent wrapping class for the child traffic channel classes. Used to allow for easy grouping of child classes in
    burst building.
    '''
    n = 1

    def __init__(self, n: int = 1):
        self.channel_type = ChannelKind.TRAFFIC_TYPE
        self.n = n
        self.k5 = 432
        if self.n not in [1, 2, 4, 8]:
            raise ValueError(f"The passed n - interleaving value of {self.n} is not valid.")


class TCH_S(TrafficChannel):  # pylint: disable=invalid-name
    '''
    Speech Traffic Channel (TCH/S)

    The traffic channels shall carry user information, defined for speech.
    '''
    slot_length = ""

    def __init__(self, slot_length: Literal[SlotLength.FULL_SUBSLOT] |
                 Literal[SlotLength.HALF_SUBSLOT] = SlotLength.FULL_SUBSLOT,
                 n: int = 1):
        super().__init__(n)
        self.slot_length = slot_length
        if self.slot_length not in (SlotLength.HALF_SUBSLOT, SlotLength.FULL_SUBSLOT):
            raise ValueError(f"The passed slot length value of {self.slot_length} is not of: 'half' or 'full' ")
        if self.n not in [1]:
            raise ValueError(f"The passed n - interleaving value of {self.n}"
                             f" is not valid for {self.__class__.__name__}")
        self.k1 = 432 if self.slot_length == SlotLength.FULL_SUBSLOT else 216
        self.k5 = 432 if self.slot_length == SlotLength.FULL_SUBSLOT else 216
        self.channel = ChannelName.TCH_S_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_1_blocks = input_data_blocks.copy()
        self.validate_k_length(1)
        if self.slot_length == SlotLength.FULL_SUBSLOT:
            self.type_4_blocks = input_data_blocks
            self.type_5_blocks = empty(shape=(self.m, 432), dtype=uint8)
            for i in range(self.m):
                self.type_5_blocks[i] = scrambler(self.type_4_blocks[i])

        elif self.slot_length == SlotLength.HALF_SUBSLOT:
            self.type_3_blocks = input_data_blocks
            self.type_4_blocks = empty(shape=(self.m, 216), dtype=uint8)
            self.type_5_blocks = empty(shape=(self.m, 216), dtype=uint8)
            for i in range(self.m):
                self.type_4_blocks[i] = block_interleaver(self.type_3_blocks[i], 101)
                self.type_5_blocks[i] = scrambler(self.type_4_blocks[i])

        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_5_blocks = input_data_blocks.copy()
        self.validate_k_length(5)
        self.type_4_blocks = empty(shape=(self.m, self.k5), dtype=uint8)
        if self.slot_length == SlotLength.HALF_SUBSLOT:
            self.type_3_blocks = empty(shape=(self.m, self.k1), dtype=uint8)
        for i in range(self.m):
            self.type_4_blocks[i] = descrambler(self.type_5_blocks[i])
            if self.slot_length == SlotLength.HALF_SUBSLOT:
                self.type_3_blocks[i] = block_deinterleaver(self.type_4_blocks[-1], 101)

        if self.slot_length == SlotLength.FULL_SUBSLOT:
            self.type_1_blocks = self.type_4_blocks
        elif self.slot_length == SlotLength.HALF_SUBSLOT:
            self.type_1_blocks = self.type_3_blocks

        self.validate_k_length(1)

    def steal_block_a(self):
        # in the case we are allocating bursts and we must steal a traffic channel with a full slot TCH/S,
        # we must remap into a half one
        # this method just remaps an existing full slot TCH/S into a halve one, discarding block A bits
        self.slot_length = SlotLength.HALF_SUBSLOT
        self.k1 = 216
        self.k5 = 216
        self.type_1_blocks[0][:216] = self.type_1_blocks[0][216:432]
        self.type_1_blocks.resize(1, 216)
        self.encode_type5_bits(self.type_1_blocks)


class TCH_7_2(TrafficChannel):  # pylint: disable=invalid-name
    '''
    7,2 kbit/s net rate (TCH/7.2)

    The traffic channels shall carry user information. Different traffic channels are defined for
    speech or data applications and for different data message speeds
    '''
    def __init__(self, n: int = 1):
        super().__init__(n)
        if self.n not in [1]:
            raise ValueError(f"The passed n - interleaving value of {self.n}"
                             f" is not valid for {self.__class__.__name__}")
        self.k1 = 432
        self.channel = ChannelName.TCH_7_2_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)
        self.type_4_blocks = self.type_1_blocks

        self.type_5_blocks = empty(shape=(self.m, self.k5), dtype=uint8)
        for i in range(self.m):
            self.type_5_blocks[i] = scrambler(self.type_4_blocks[i])

        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_4_blocks = empty(shape=(self.m, self.k5), dtype=uint8)

        for i in range(self.m):
            self.type_4_blocks[i] = descrambler(self.type_5_blocks[i])

        self.type_1_blocks = self.type_4_blocks
        self.validate_k_length(1)


class TCH_4_8(TrafficChannel):  # pylint: disable=invalid-name
    '''
    4,8 kbit/s net rate (TCH/4.8)

    The traffic channels shall carry user information. Different traffic channels are defined for
    speech or data applications and for different data message speeds. Interleaving of depths n = 1,4, or 8 possible.
    '''
    def __init__(self, n: int = 1):
        super().__init__(n)
        if self.n not in [1, 4, 8]:
            raise ValueError(f"The passed n - interleaving value of {self.n}"
                             f" is not valid for {self.__class__.__name__}")
        self.k1 = 288
        self.channel = ChannelName.TCH_4_8_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)
        self.type_2_blocks = empty(shape=(self.m, 292), dtype=uint8)
        self.type_3_blocks = empty(shape=(self.m, 432), dtype=uint8)
        for i in range(self.m):
            self.type_2_blocks[i][:-4] = self.type_1_blocks[i]
            self.type_2_blocks[i][-4:] = LOGICAL_CH_TAIL_BITS
            self.type_3_blocks[i] = rcpc_encoder(self.type_2_blocks[i], 292, 432)

        self.type_4_blocks = n_block_interleaver(self.type_3_blocks, self.n)
        self.type_5_blocks = empty(shape=((self.m+self.n-1), 432), dtype=uint8)

        # blocks are scrambled individually
        for i in range((self.m+self.n-1)):
            self.type_5_blocks[i] = scrambler(self.type_4_blocks[i])

        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = len(input_data_blocks) + 1 - self.n
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)

        self.type_4_blocks = empty(shape=((self.m+self.n-1), 432), dtype=uint8)
        # blocks are scrambled individually
        for i in range((self.m+self.n-1)):
            self.type_4_blocks[i] = descrambler(self.type_5_blocks[i])

        self.type_3_blocks = n_block_deinterleaver(self.type_4_blocks, self.m, self.n)

        self.type_2_blocks = empty(shape=(self.m, 292), dtype=uint8)
        self.type_1_blocks = empty(shape=(self.m, 288), dtype=uint8)

        for i in range(self.m):
            self.type_2_blocks[i] = rcpc_decoder(self.type_3_blocks[i], 292, 292, 432)
            self.type_1_blocks[i] = self.type_2_blocks[i][:-4]

        self.validate_k_length(1)


class TCH_2_4(TrafficChannel):  # pylint: disable=invalid-name
    '''
    2,4 kbit/s net rate (TCH/2.4)

    The traffic channels shall carry user information. Different traffic channels are defined for
    speech or data applications and for different data message speeds. Interleaving of depths n = 1,4, or 8 possible.
    '''
    def __init__(self, n: int = 1):
        super().__init__(n)
        if self.n not in [1, 4, 8]:
            raise ValueError(f"The passed n - interleaving value of {self.n}"
                             f" is not valid for {self.__class__.__name__}")
        self.k1 = 144
        self.channel = ChannelName.TCH_2_4_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0]
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)

        self.type_2_blocks = empty(shape=((self.m+self.n-1), 148), dtype=uint8)
        self.type_3_blocks = empty(shape=((self.m+self.n-1), 432), dtype=uint8)
        for i in range(self.m):
            self.type_2_blocks[i][:-4] = self.type_1_blocks[i]
            self.type_2_blocks[i][-4:] = LOGICAL_CH_TAIL_BITS
            self.type_3_blocks[i] = rcpc_encoder(self.type_2_blocks[i], 148, 432)

        self.type_4_blocks = n_block_interleaver(self.type_3_blocks, self.n)
        self.type_5_blocks = empty(shape=((self.m+self.n-1), 432), dtype=uint8)

        # blocks are scrambled individually
        for i in range((self.m+self.n-1)):
            self.type_5_blocks[i] = scrambler(self.type_4_blocks[i])

        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        self.m = input_data_blocks.shape[0] + 1 - self.n
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)

        self.type_4_blocks = empty(shape=((self.m+self.n-1), 432), dtype=uint8)
        # blocks are scrambled individually
        for i in range((self.m+self.n-1)):
            self.type_4_blocks[i] = descrambler(self.type_5_blocks[i])

        self.type_3_blocks = n_block_deinterleaver(self.type_4_blocks, self.m, self.n)

        self.type_2_blocks = empty(shape=(self.m, 148), dtype=uint8)
        self.type_1_blocks = empty(shape=(self.m, 144), dtype=uint8)

        for i in range(self.m):
            self.type_2_blocks[i] = rcpc_decoder(self.type_3_blocks[i], 148, 148, 432)
            self.type_1_blocks[i] = self.type_2_blocks[i][:-4]

        self.validate_k_length(1)


###################################################################################################
# Linearization Channels

class LinearizationChannel(LogicalChannelVD):
    '''
    Parent wrapping class for the child linearization channel classes. Used to allow for easy grouping of child
    classes in burst building.
    '''
    def __init__(self):
        self.channel_type = ChannelKind.LINEARIZATION_TYPE


###################################################################################################
class CLCH(LinearizationChannel):
    '''
    Common Linearization CHannel (CLCH)

    up-link, shared by all the MSs;
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 206
        self.k5 = 206
        self.m = 1
        self.channel = ChannelName.CLCH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Simply copies 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks directly, no
        encoding is performed.

        :param self: CLCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 206, but with only 1 block/row
         to be copied into self.type_5_blocks of length 206
        :type input_data_blocks: NDArray[uint8]
        """
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)
        self.type_5_blocks = self.type_1_blocks
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Simply copies 1 block of type-5 bits from input_data_blocks into 1 block within self.type_1_blocks directly, no
        encoding is performed.

        :param self: CLCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 206, but with only 1 block/row
         to be copied into self.type_1_blocks of length 206
        :type input_data_blocks: NDArray[uint8]
        """
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_1_blocks = self.type_5_blocks
        self.validate_k_length(1)


class BLCH(LinearizationChannel):
    '''
    BS Linearization CHannel (BLCH)

    downlink, used by the BS
    '''
    def __init__(self):
        super().__init__()
        self.k1 = 216
        self.k5 = 216
        self.m = 1
        self.channel = ChannelName.BLCH_CHANNEL

    def encode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Simply copies 1 block of type-1 bits in input_data_blocks into 1 block within self.type_5_blocks directly, no
        encoding is performed.

        :param self: BLCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 216, but with only 1 block/row
         to be copied into self.type_5_blocks of length 216
        :type input_data_blocks: NDArray[uint8]
        """
        self.type_1_blocks = input_data_blocks
        self.validate_k_length(1)
        self.type_5_blocks = self.type_1_blocks
        self.validate_k_length(5)

    def decode_type5_bits(self, input_data_blocks: NDArray[uint8]):
        """
        Simply copies 1 block of type-5 bits from input_data_blocks into 1 block within self.type_1_blocks directly, no
        encoding is performed.

        :param self: BLCH logical channel
        :param input_data_blocks: 2 dimensional input binary uint8 array of length 216, but with only 1 block/row
         to be copied into self.type_1_blocks of length 216
        :type input_data_blocks: NDArray[uint8]
        """
        self.type_5_blocks = input_data_blocks
        self.validate_k_length(5)
        self.type_1_blocks = self.type_5_blocks
        self.validate_k_length(1)
