"""
coding_scrambling.py contains the CRC, interleaving, encoding, and scrambling methods and their respective inverses for
converting type 1 bits into type 5 bits for logical channels.
"""
from math import gcd
from collections import deque
from numpy import zeros, full, uint32, argmin, uint8, array, arange, empty, broadcast_to, asarray, bitwise_or, \
    bitwise_xor, bitwise_count, flatnonzero, int32, clip
from numpy.typing import NDArray

# MNC, MCC, and Colour code are chosen at random here to feed the seed for the scrambler
MCC = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]                # order is MSB -> LSB
MNC = [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0]    # order is MSB -> LSB
COLOUR_CODE = [1, 0, 1, 0, 1, 0]                    # order is MSB -> LSB

N_BLOCK_BIT_LENGTH = 432
N_BLOCK_BLOCK_INTERLEAVER_A_VALUE = 103
INF = 10**9

RM_30_14_GENERATOR = array([0b100000000000001001101101100000,
                            0b010000000000000010110111100000,
                            0b001000000000001111110000100000,
                            0b000100000000001110000000111100,
                            0b000010000000001001100000111010,
                            0b000001000000000101010000110110,
                            0b000000100000000010110000101110,
                            0b000000010000001111111111011111,
                            0b000000001000001000001100111001,
                            0b000000000100000100001010110101,
                            0b000000000010000010000110101101,
                            0b000000000001000001001001110011,
                            0b000000000000100000100101101011,
                            0b000000000000010000010011100111], dtype=uint32)

# constant used to shift 32 bit words down to 1 bit for calculation of RM(30,14) encoding
_RM_ENCODER_SHIFTS_30 = arange(29, -1, -1, dtype=uint32)
_RM_DECODER_SHIFTS_14 = arange(13, -1, -1, dtype=uint32)

# generator polynominal is ordered as Gn: [d0(current), d1, d2, d3, d4], for [G1, G2, G3, G4] overall
RCPC_MOTHER_GENERATOR = array([[1, 1, 0, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 0, 1, 1]], dtype=uint8)
RCPC_TAIL_LENGTH = 4
RCPC_TRACEBACK_LENGTH = 40   # instead of performing full Viterbi traceback we just limit here to reasonable amount

TETRA_32_SCRAMBLING_SEQUENCE = array([0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 16, 22, 23, 26, 31], dtype=uint8)

###################################################################################################


def _build_rm_codebook() -> NDArray[uint32]:
    """Initializes a codebook containing all possible 2^14 codewords for decoding (30,14) RM encoded data.

    Returns:

        NDArray[uint32]: Entire 2^14 codeword codebook for the Tetra Shorted Reed Muller (30,14) code
    """
    # 1. Generate 2^14 gray code input_data possibilities

    # gray code generation inner function is taken from avanitrachhadiya2155
    n = 1 << 14

    i = arange(n, dtype=uint32)
    gray_binary_values = i ^ (i >> uint32(1))

    # 2. Calculate the resulting codewords, note that due to gray code input_data, only one bit changes at a time,
    #    this greatly simplifies the calculation and also makes searching the code book very basic.

    # first input_data and output codeword is all 0's
    codebook = zeros(2**14, dtype=uint32)
    prev_g = int(gray_binary_values[0])
    codeword = uint32(0)
    codebook[prev_g] = codeword

    for i in range(1, 2**14):  # type: ignore[assignment]
        g = int(gray_binary_values[i])
        # because the RM(30,14) encoding is linear, can rapidly calculate next codeword if we know the difference
        # between the previous input_data and the current word
        delta = g ^ prev_g
        # determine which bit changed -> this coorelates to the row in the generator matrix
        bit_pos = delta.bit_length() - 1
        generator_index = 13 - bit_pos
        # only need to perform one calculation knowing the applicable row of the generator matrix to use
        codeword ^= uint32(RM_30_14_GENERATOR[generator_index])
        codebook[g] = codeword
        prev_g = g

    return codebook


RM_30_14_CODEBOOK = _build_rm_codebook()

###################################################################################################


def _build_rcpc_transition_table() -> dict[int, list[tuple[int, int, list[int]]]]:
    """Builds a RCPC convolution state transition table dict for Viterbi decoding. The dictionary is in the form of a
    16-key dictionary of destination states with values for origin RCPC stage, input bit and, list of 4 output bits

    Returns:
        dict[int,list[tuple[int,int,list[int]]]]: keys of dest. RCPC states, with values of
        [origin RCPC state, input bit, four output bits[1,2,3,4]]
    """
    temp_table: dict
    temp_table = {i: [] for i in range(16)}  # each key is one of the 16 states,

    states = [[(i >> 3) & 0b1, (i >> 2) & 0b1, (i >> 1) & 0b1, i & 0b1] for i in range(16)]

    # for each of the states, calculate the possible destinations using 0 or 1 as input_data transistion,
    # then log the result, indexed by destination
    for state_bits in states:
        for input_bit in [0, 1]:
            input_data = [input_bit] + state_bits
            # add input_data to shift register, i.e. the state, for easy XOR'ing with generator polynominal
            output_bits = [0, 0, 0, 0]  # order c1, c2, c3, c4
            for i in range(4):
                # for i e [0,4] calculate the c value, recall for every 1 bit input_data we output 5 bits
                c = 0
                for j in range(5):
                    # to generate output data simple XOR operation using generator polynomial as a bit mask
                    c ^= (input_data[j] & RCPC_MOTHER_GENERATOR[i][j])
                output_bits[i] = c

            # destination state = shift register after input_data
            destination_state_bits = [input_bit] + state_bits[:-1]
            # for simplicity of lookup we just store the states as integers 0-15
            current_state = sum(x << (3-i) for i, x in enumerate(state_bits))
            destination_state = sum(x << (3-i) for i, x in enumerate(destination_state_bits))
            temp_table[destination_state].append((current_state, input_bit, output_bits))

    # for key in temp_table.keys():
    #     print(f"Destination State   Input Bit  Output Bits     Source State")
    #     print(f"{key} \t\t\t {temp_table[key][0][1]} \t {temp_table[key][0][2]} \t\t {temp_table[key][0][0]}")
    #     print(f"{key} \t\t\t {temp_table[key][1][1]} \t {temp_table[key][1][2]} \t\t {temp_table[key][1][0]}")

    return temp_table


RCPC_TRANSITION_TABLE = _build_rcpc_transition_table()

###################################################################################################
# CRC-16 / Block Encoding (type 1 - to - type 2 bits)


def crc16_encoder(input_data: NDArray[uint8]) -> NDArray[uint8]:
    """Encodes K # of input_data bits into (K+16) # of output bits in the form [K-Data Bits | 16-CRC bits]

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array

    Returns:
        NDArray[uint8]: output binary values stored in uint8 bytes array
    """
    # The first K bits are just copies of the input_data
    k_len = input_data.size
    encoded_bits = empty(shape=(k_len+16), dtype=uint8)
    encoded_bits[:-16] = input_data

    # For the last 16 bits, we need to calculate the CRC
    r = 0x0000
    for k in range(k_len):
        msb = (r >> 15) & 1
        fb = msb ^ int(input_data[k])
        r = (r << 1) & 0xFFFF
        if fb == 1:
            r ^= 0x1021

    # After running through K bits, we flush the CRC registers with 16 0's
    for _ in range(16):
        msb = (r >> 15) & 1
        r = (r << 1) & 0xFFFF
        if msb == 1:
            r ^= 0x1021

    for i in range(16):
        encoded_bits[k_len+i] = (r >> (15-i)) & 1

    return encoded_bits

###################################################################################################


def crc16_decoder(input_data: NDArray[uint8]) -> tuple[NDArray[uint8], bool]:
    """Decodes K # of input_data bits from (K+16) # of input_data bits in the form [K-Data Bits | 16-CRC bits]
    and determines if the rx input_data data CRC matches the calculated crc from the rx payload data

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be decoded

    Returns:
        tuple[NDArray[uint8], bool]: (decoded output binary values stored in uint8 bytes array, if rx crc matches
        the crc value present in the input data appended)
    """
    # The first K bits are just copies of the payload
    k_len = input_data.size-16
    decoded_bits = empty(shape=k_len, dtype=uint8)
    decoded_bits[:] = input_data[0:k_len]

    # Then, we recalculate the CRC to compare against the last 16 bits
    crc_bits = input_data[k_len:k_len+16]
    r = 0x0000
    for k in range(k_len):
        msb = (r >> 15) & 1
        fb = msb ^ int(input_data[k])
        r = (r << 1) & 0xFFFF
        if fb == 1:
            r ^= 0x1021

    # After running through K bits, we flush the CRC registers with 16 0's
    for _ in range(16):
        msb = (r >> 15) & 1
        r = (r << 1) & 0xFFFF
        if msb:
            r ^= 0x1021

    # Convert rx CRC value to a binary value to compare against the calculated one
    rx_crc = 0
    for i in range(16):
        rx_crc = (rx_crc << 1) | int(crc_bits[i])

    # Determine if the match, return the result packed with the decoded bit array
    crc_valid = rx_crc == r

    return decoded_bits, crc_valid


def rm3014_encoder(input_data: NDArray[uint8]) -> NDArray[uint8]:
    """Encodes 14 input_data bits into 30 shorted (30,14) Reed Muller encoded output bits

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be encoded

    Returns:
        NDArray[uint8]: encoded output binary values stored in uint8 bytes array
    """
    # encoding is very simple, input_data vector (1,14) XORs with
    # (14,30) generator matrix 'RM_30_14_GENERATOR' to yield (1,14) length output vector
    # to speed up calculatations instead of bit arrays we just use binary values

    bits = asarray(input_data, dtype=uint8).reshape(14)

    # calculate the mask which basically forms the XOR'ing pattern required
    mask = (bits & 1).astype(bool)
    codeword_output = bitwise_xor.reduce(RM_30_14_GENERATOR[mask], dtype=uint32, initial=uint32(0))

    # shift 32 bit results down and mask with 1 to get the resulting bit
    return ((codeword_output >> _RM_ENCODER_SHIFTS_30) & 1).astype(uint8)


def rm3014_decoder(input_data: NDArray[uint8]) -> NDArray[uint8]:
    """Decodes 30 (30,14) Reed Muller encoded input_data bits into 14 output bits,
    performing hard minimum distance decoding using an LUT codebook.

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be decoded

    Returns:
        NDArray[uint8]: decoded output binary values stored in uint8 bytes array
    """
    # There are papers on majority logic algorithm for shortned RM codes for TETRA, but I don't have access
    # and because of the nature of the usage of the RM(30,14) on access channels without any additional
    # convolutional encoding and only being 14 bits, it's more sure-fire to just use hard table decoding
    # instead of attempting syndrome decoding of some sort, tba perhaps

    # 1. Convert input_data bit array into a 30 bit binary int value
    # convert to uint32
    input_uint32 = (asarray(input_data, dtype=uint8).reshape(30) & 1).astype(uint32)
    # convert into single value using or's, using the precalculate shift vector to align MSB->LSB
    input_word = bitwise_or.reduce(input_uint32 << _RM_ENCODER_SHIFTS_30, dtype=uint32, initial=uint32(0))

    # 2. Evaluate over the generated codebook, calculating hamming distance to determine optimal input_data word

    # evaluate the comparisons, if the word exists then we have 1 true result
    code_comparisions = flatnonzero(RM_30_14_CODEBOOK == input_word)

    if code_comparisions.size:
        closest_word = int(code_comparisions[0])
    else:
        # compute hamming distances instead over entire codebook
        x = RM_30_14_CODEBOOK ^ input_word
        d = bitwise_count(x).astype(uint8)
        closest_word = int(argmin(d))  # select closest code work as most likely

    # convert 32 bit codeword into bit array, by shifting and masking with, convert to uint8
    decoded_bits = ((uint32(closest_word) >> _RM_DECODER_SHIFTS_14) & 1).astype(uint8)

    return decoded_bits


###################################################################################################
# Convolutional encoding/decoding (type 2 - to - type 3 bits)

def _fetch_rcpc_parameters(k2: int, k3: int) -> tuple[list[int], int, int]:
    """Returns the corresponding puncturing vector, t-value, and division ratio for a given TETRA RCPC (K2,K3) ratio

    Args:
        k2 (int): input_data rate ratio numerator
        k3 (int): output rate ratio denominator

    Returns:
        tuple[list[int],int,int]: (puncturing vector int list[ ], number of puncturing vector points, division ratio)
    """
    punc_vector = []
    t = 0
    div_ratio = 0
    match (k2, k3):
        case(2, 3):
            punc_vector = [0, 1, 4]
            t = 3
        case(1, 3):
            punc_vector = [0, 1, 2, 4, 5, 6]
            t = 6
        case(292, 432):
            punc_vector = [0, 1, 4]
            t = 3
            div_ratio = 65
        case(148, 432):
            punc_vector = [0, 1, 2, 4, 5, 6]
            t = 6
            div_ratio = 35
        case _:
            raise RuntimeError(f"Passed k2, k3 pair value of: ({k2},{k3}) invalid")
    return punc_vector, t, div_ratio


def rcpc_encoder(input_data: NDArray[uint8], k2: int, k3: int) -> NDArray[uint8]:
    """Performs 16-state rate-compatible punctured convoltion (RCPC) encoding of rate (K2/K3) on the input_data data

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be encoded
        k2 (int): input_data rate ratio numerator
        k3 (int): output rate ratio denominator

    Returns:
        NDArray[uint8]: encoded output input binary values stored in uint8 bytes array
        with length (input_data)*K3 // K2
    """
    # TODO: Convert rcpc encoder to numpy
    input_data = input_data.tolist()
    # determine if combination of K2 and K3 are numerical valid given the length of the input_data data stream
    if (len(input_data) * k3) % k2 != 0:
        raise ValueError(f"RCPC encoder value of {k2}/{k3} is not valid with data length {len(input_data)}")

    # Step 1: encoding with 16-state mother code of rate 1/4
    encoded_data_temp = []

    # initialize the shift register as all 0's, order of register is d1, d2, d3, d4
    shift_register = deque([0 for _ in range(4)], maxlen=4)

    for bit in input_data:
        # add input_data to shift register for easy XOR'ing with generator polynominal
        input_temp = [bit] + list(shift_register)
        nibble_temp = [0, 0, 0, 0]  # order c1, c2, c3, c4
        for i in range(4):
            # for i e [0,4] calculate the c value, recall for every 1 bit input_data we output 5 bits
            c = 0
            for j in range(5):
                # to generate output data simple XOR operation using generator polynomial as a bit mask
                c ^= (input_temp[j] & RCPC_MOTHER_GENERATOR[i][j])
            nibble_temp[i] = c

        # add the 4 new bits
        encoded_data_temp.extend(nibble_temp)
        # add new data point to top of shift reg, discard oldest bit on rightside
        shift_register.appendleft(bit)

    # Step 2: puncturing of the mother code to obtain a 16-state RCPC code fo rate K2/K3
    #         with possible rates of: 2/3, 1/3, 292/432, 148/432

    punc_vector, t, div_ratio = _fetch_rcpc_parameters(k2, k3)
    punctured_data = []

    num_output_bits = (len(input_data) * k3) // k2
    for j in range(num_output_bits):
        # j represents output data index
        i = j if div_ratio == 0 else (j + (j // div_ratio))
        # i indexes the puncturing pattern
        k = 8*(i // t) + punc_vector[i % t] # P indicing equivalent to i - t(i//t) for integers
        # k indexes the mother-code bitstream

        punctured_data.append(encoded_data_temp[k])

    return array(punctured_data, dtype=uint8)


def rcpc_decoder(input_data: NDArray[uint8], payload_length: int, k2: int, k3: int) -> NDArray[uint8]:
    """Performs Viterbi decoding of 16-state RCPC convolutional encoded data of rate (K2/K3)
    into decoded bits into an array of 'payloadLength' length

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be decoded
        payload_length (int): int representing the number of output payload bits expected
        k2 (int): input_data rate ratio numerator
        k3 (int): output rate ratio denominator

    Returns:
        NDArray[uint8]: decoded output input binary values stored in uint8 bytes array of length 'payload_length'
    """
    # TODO: convert rcpcDecoder to numpy
    input_data = input_data.tolist()

    decoded_bits = []
    # Determine the t, divRatio, and puncturing index values based on K2,K3
    punc_vector, t, div_ratio = _fetch_rcpc_parameters(k2, k3)

    # This Viterbi decoder uses 1 data bit = 1 Viterbi transition basis
    # This results in some complication because, due to fractional puncturing,
    # some transitions are erased from the input_data encoded stream, which is based on a 16-state
    # However, because we know the encoding scheme, we can calculate just in-time which
    # bits of the mothercode we must consider, some 4-bit pairs we can just ignore

    # inner function for calculating output data bit index, k, and i values for puncturing
    def map_r_to_u_b(r_idx):
        i = r_idx if div_ratio == 0 else r_idx + (r_idx // div_ratio)
        k = 8 * (i // t) + punc_vector[i % t]
        return (k // 4), (k % 4)

    # Path metrics
    path_metric = full(16, INF, dtype=int)
    # initial decoder state is set to 0,0,0,0 just as decoder performs
    path_metric[0] = 0

    # Use a streaming linear register method for decoding to reduce memory depth required,
    # but requires flushing at the end of input_data stream
    survivor_bits: list
    survivor_bits = [deque(maxlen=RCPC_TRACEBACK_LENGTH) for _ in range(16)]

    j = 0
    pending: dict
    pending = {}  # u -> list[(b, rxBit)] if we ever read ahead such as when j // divRatio increments

    # Iterate over each input_data data payload bit
    for u in range(payload_length):

        # 1. Calculate the bit puncturing values for the observation of the mother code
        obs_u = pending.pop(u, []) if u in pending else []
        # Iterate over the data length to determine for input_data data bit / transistion [u]
        # what mother code bits are used [b_r]
        while j < len(input_data):
            # calculate output bit/transition index u_r and puncture index b_r
            u_r, b_r = map_r_to_u_b(j)
            # if we have a recieve bit that lies as part of next transition due to skips, store it
            if u_r > u:
                # cache for later and stop consuming for this u
                pending.setdefault(u_r, []).append((b_r, input_data[j]))
                j += 1
                continue
            # u_r == u
            obs_u.append((b_r, input_data[j]))
            j += 1
        obs_u.sort(key=lambda x: x[0])

        # Note for the last 4 output data bits, we know that they are set to 0,0,0,0
        tail_enforced = u >= payload_length - RCPC_TAIL_LENGTH
        # Temp. update metrics
        new_path_metric = full(16, INF, dtype=int)
        new_survivor_bits: list
        new_survivor_bits = [deque(maxlen=RCPC_TRACEBACK_LENGTH) for _ in range(16)]

        # for each state, determine most likely possible transistion to it
        for dest_state in range(16):
            most_likely_metric = INF
            most_likely_origin_state = -1
            most_likely_input_bit = 0
            # for each state determine the most likely transition by finding the minimum path length to it
            for (origin_state, input_bit, output_bits) in RCPC_TRANSITION_TABLE[dest_state]:
                if tail_enforced and input_bit != 0:
                    continue
                metric = path_metric[origin_state]
                metric += sum(rxBit ^ output_bits[b] for (b, rxBit) in obs_u)
                if metric < most_likely_metric:
                    most_likely_metric = metric
                    most_likely_origin_state = origin_state
                    most_likely_input_bit = input_bit

            # Find most likely path to current state considered
            new_path_metric[dest_state] = most_likely_metric
            if most_likely_origin_state >= 0 and most_likely_metric < INF:
                new_survivor_bits[dest_state] = deque(survivor_bits[most_likely_origin_state],
                                                      maxlen=RCPC_TRACEBACK_LENGTH)
                new_survivor_bits[dest_state].append(most_likely_input_bit)

        # update overall pathMetric and the survivor bit streams
        path_metric = new_path_metric
        survivor_bits = new_survivor_bits

        # stream output once we have evaluated atleast RCPC_TRACEBACK_LENGTH-1 # of recieved bits
        if u >= RCPC_TRACEBACK_LENGTH - 1:
            most_likely_state = int(argmin(path_metric))
            # oldest bit in that survivor corresponds to (u - (l-1))
            decoded_bits.append(survivor_bits[most_likely_state][0])
            for s in range(16):
                if survivor_bits[s]:
                    survivor_bits[s].popleft()

    # the last state terminates to zero, data-aided, just flush the remaining values out
    final_state = int(argmin(path_metric))
    remaining = list(survivor_bits[final_state])
    decoded_bits.extend(remaining)

    return array(decoded_bits, dtype=uint8)

###################################################################################################
# Interleaving/deinterleaving (type 3 - to - type 4 bits)


def block_interleaver(input_data: NDArray[uint8], a: int) -> NDArray[uint8]:
    """Performs (K,a) block interleaving of K input_data data into K output data

    Args:
        input_data (NDArray[uint8]): block of input binary values stored in uint8 bytes array to be interleaved
        a (int): interleaving coefficent

    Returns:
        NDArray[uint8]: interleaved block of output binary values stored in uint8 bytes array
    """
    # length of output interleaved data is the same as the input_data data: K
    k = input_data.size

    interleaved_data = empty(shape=k, dtype=uint8)
    _, index = _block_perm(k, a, True)
    interleaved_data[index] = input_data

    return interleaved_data


def block_deinterleaver(input_data: NDArray[uint8], a: int) -> NDArray[uint8]:
    """Performs (K,a) block deinterleaving of K input_data data into K output data

    Args:
        input_data (NDArray[uint8]): block of input binary values stored in uint8 bytes array to be deinterleaved
        a (int): interleaving coefficent

    Returns:
        NDArray[uint8]: deinterleaved block of output binary values stored in uint8 bytes array
    """
    k = input_data.size

    deinterleaved_data = empty(shape=k, dtype=uint8)
    index, _ = _block_perm(k, a, False)
    deinterleaved_data[index] = input_data

    return deinterleaved_data


def _diagonal_mapping(n: int):
    if n not in [1, 4, 8]:
        raise ValueError(f"Invalid {n}, expected value in [1, 4, 8]")
    w = N_BLOCK_BIT_LENGTH // n  # blockwidth

    k = arange(N_BLOCK_BIT_LENGTH, dtype=int32)
    j = k // w
    i = k % w

    src_index = j + (i * n)
    return j, src_index


def _block_perm(k_len: int, a: int, interleave: bool = True):
    if gcd(a, k_len) != 1:
        raise ValueError(f"{k_len} and {a} does not have a greatest common divisor of 1"
                         f", therefore cannot perform (K,a) deinterleaving")

    i = arange(k_len, dtype=int32)
    if not interleave:
        p = (a * i) % k_len
        p_inv = None  # don't bother finding
    else:
        p = None
        a_inv = pow(a, -1, k_len)
        p_inv = (a_inv * i) % k_len

    return p, p_inv


def n_block_interleaver(input_data_blocks: NDArray[uint8], n: int) -> NDArray[uint8]:
    """Performs n block interleaving of m # of input_data data blocks into (m+n-1) # of output data blocks

    Args:
        input_data_blocks (NDArray[uint8]): 2-dimen. array of blocks with binary values stored in uint8 bytes arrays
        n (int): int representing the number of blocks to interleave: (1,4,8)

    Returns:
        NDArray[uint8]: 2-dimen. output array of blocks with binary values stored in uint8 bytes arrays
    """
    # n may be 1, 4, or 8, interleaving m # of 432 bit long blocks over n blocks into a sequence of (m + n - 1) blocks

    if input_data_blocks.ndim != 2 or input_data_blocks.shape[1] != N_BLOCK_BIT_LENGTH:
        raise ValueError(f"Expected shape of (m, {N_BLOCK_BIT_LENGTH}, got {input_data_blocks.shape}")

    m = input_data_blocks.shape[0]
    l_len = m + n - 1  # Number of output blocks

    # Step 1, diagonal interleave m blocks in (m+n-1) blocks, each with 432 bits in them
    j_of_k, src_index = _diagonal_mapping(n)
    m_array = arange(l_len, dtype=int32)[:, None]   # (l, 1)
    src_block = m_array - j_of_k[None, :]       # (l, 432)

    # valid positions to prevent out of bounds indexing, instead we leave those bits as zero from the init
    valid = (src_block >= 0) & (src_block < m)

    src_block_clip = clip(src_block, 0, m-1)

    temp = input_data_blocks[src_block_clip, src_index[None, :]]
    temp = temp.astype(uint8, copy=False)
    temp[~valid] = 0  # set unaccessed bits to zero

    # Step 2, block interleave
    _, p_inv = _block_perm(N_BLOCK_BIT_LENGTH, N_BLOCK_BLOCK_INTERLEAVER_A_VALUE, True)
    interleaved_data_blocks = temp[:, p_inv]

    return interleaved_data_blocks


def n_block_deinterleaver(input_data_blocks: NDArray[uint8], m: int, n: int) -> NDArray[uint8]:
    """Performs n block deinterleaving of (m+n-1) # of input_data data blocks into  # of output data blocks

    Args:
        input_data_blocks (NDArray[uint8]): 2-dimen. array of blocks with binary values stored in uint8 bytes arrays
        m (int): int representing the number of original input blocks expected to deinterleave
        n (int): int representing the number of blocks to deinterleave: (1,4,8)

    Returns:
        NDArray[uint8]: 2-dimen. deinterleaved output array of blocks with binary values stored in uint8 bytes arrays
    """
    if input_data_blocks.ndim != 2 or input_data_blocks.shape[1] != N_BLOCK_BIT_LENGTH:
        raise ValueError(f"Expected shape of (m, {N_BLOCK_BIT_LENGTH}, got {input_data_blocks.shape}")

    l_len = input_data_blocks.shape[0]

    if l_len != (m + n - 1):
        raise ValueError(f"Expected {m+n-1} blocks, got {l_len}")

    # Step 1, block deinterleave
    p, _ = _block_perm(N_BLOCK_BIT_LENGTH, N_BLOCK_BLOCK_INTERLEAVER_A_VALUE, False)
    temp = input_data_blocks[:, p]

    # Step 2, reverse diagonal interleaving of m blocks
    j_of_k, src_index = _diagonal_mapping(n)
    m_array = arange(l_len, dtype=int32)[:, None]      # (l,1)
    dest_block = m_array - j_of_k[None, :]         # (l,432)
    valid = (dest_block >= 0) & (dest_block < m)

    dest_block_clip = clip(dest_block, 0, m-1)
    deinterleaved_data_blocks = zeros((m, N_BLOCK_BIT_LENGTH), dtype=uint8)
    columns = broadcast_to(src_index[None, :], (l_len, N_BLOCK_BIT_LENGTH))

    # only assigned valid positions
    deinterleaved_data_blocks[dest_block_clip[valid], columns[valid]] = temp[valid]

    return deinterleaved_data_blocks

###################################################################################################
# Scrambling/descrambling (type 4 - to - type 5 bits)


def scrambler(input_data: NDArray[uint8], bsch_state: bool = False) -> NDArray[uint8]:
    """Performs scrambling of input_data data using a predefined scrambling polynominal

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be scrambled
        bsch_state (bool, optional): Defaults to False, if the logical channel
        is BSCH, should set to True to set scramlber init to 0's

    Returns:
        NDArray[uint8]: scrabled output input binary values stored in uint8 bytes array
    """
    n = input_data.size
    scrambled_data = zeros(shape=input_data.size, dtype=uint8)
    scrambler_init_code = []

    if not bsch_state:
        scrambler_init_code = [1, 1] + MCC + MNC + COLOUR_CODE
    else:
        # for BSCH with scramble with zeros
        scrambler_init_code = [1, 1] + [0]*30

    shift_register = deque(scrambler_init_code, maxlen=32)
    for i in range(n):
        p_k = uint8(shift_register[31])  # oldest data

        scrambled_data[i] = input_data[i] ^ p_k

        feedback = 0
        for k in TETRA_32_SCRAMBLING_SEQUENCE:
            feedback ^= shift_register[k]

        shift_register.popleft()
        shift_register.append(feedback)

    return scrambled_data


def descrambler(input_data: NDArray[uint8], bsch_state: bool = False) -> NDArray[uint8]:
    """_summary_

    Args:
        input_data (NDArray[uint8]): input binary values stored in uint8 bytes array to be descrambled
        bsch_state (bool, optional): Defaults to False, if the logical channel
        is BSCH, should set to True to set scramlber init to 0's

    Returns:
        NDArray[uint8]: descrambled output binary values stored in uint8 bytes array
    """
    # the scrambler is self-inverse because it is just XOR with sequence, so can apply the same operation to descramble
    return scrambler(input_data, bsch_state)
