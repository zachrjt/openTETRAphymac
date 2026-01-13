# ZT - 2026
# Based on EN 300 392-2 V2.4.2
from numpy import zeros, full, uint32, argmin
from typing import List, Tuple
from math import gcd
from collections import deque

# TODO: Refactor to use numpy ndarrays to improve performance, write unit tests

# MNC, MCC, and Colour code are chosen at random here to feed the seed for the scrambler
MCC = [1,0,1,0,1,0,1,0,0,0] # order is MSB -> LSB
MNC = [1,0,1,0,1,1,1,1,0,1,1,0,1,0] # order is MSB -> LSB
COLOUR_CODE = [1,0,1,0,1,0] # order is MSB -> LSB

N_BLOCK_BIT_LENGTH = 432
N_BLOCK_BLOCK_INTERLEAVER_A_VALUE = 103
INF = 10**9

RM_30_14_GENERATOR =   [0b100000000000001001101101100000,
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
                        0b000000000000010000010011100111]

RM_30_14_codebook = None

# generator polynominal is ordered as Gn: [d0(current), d1, d2, d3, d4], for [G1, G2, G3, G4] overall
RCPC_MOTHER_GENERATOR = [[1,1,0,0,1], [1,0,1,1,1], [1,1,1,0,1], [1,1,0,1,1]]
RCPC_TAIL_LENGTH = 4
RCPC_TRACEBACK_LENGTH = 40   # instead of performing full Viterbi traceback we just limit here to reasonable amount
RCPC_transition_table = None

TETRA_32_SCRAMBLING_SEQUENCE = [0,1,2,4,5,7,8,10,11,12,16,22,23,26,31]

###################################################################################################
# CRC-16-CCITT / Block Encoding (type 1 - to - type 2 bits)

def crc16Encoder(inputData:List[int], K:int) -> List[int]:
    '''
    Encodes K # of inputData bits into (K+16) # of output bits in the form [K-Data Bits | 16-CRC bits]
    
    :param inputData: array of 1 and 0's
    :param K: length of inputData
    '''
    # The first K bits are just copies of the input
    encodedBits = inputData + [0 for _ in range(16)]

    # For the last 16 bits, we need to calculate the CRC
    R = 0x0000
    for k in range(K):
        msb = (R >> 15) & 1
        fb = msb ^ inputData[k]
        R = (R << 1) & 0xFFFF
        if fb == 1:
            R ^= 0x1021

    # After running through K bits, we flush the CRC registers with 16 0's
    for _ in range(16):
        msb = (R >> 15) & 1
        R = (R << 1) & 0xFFFF
        if fb == 1:
            R ^= 0x1021
    
    for i in range(16):
        encodedBits[K+i] = (R >> (15-i)) & 1

    return encodedBits

def crc16Decoder(inputData:List[int], K:int) -> Tuple[List[int], bool]:
    '''
    Decodes K # of inputData bits from (K+16) # of input bits in the form [K-Data Bits | 16-CRC bits]
    and determines if the rx input data CRC matches the calculated crc from the rx payload data
    
    :param inputData: array of 1 and 0's
    :param K: length of payload, (length of data - 16)
    '''
    # The first K bits are just copies of the payload
    decodedBits = inputData[0:K]

    # Then, we recalculate the CRC to compare against the last 16 bits
    crcBits = inputData[K:K+16]
    R = 0x0000
    for k in range(K):
        msb = (R >> 15) & 1
        fb = msb ^ inputData[k]
        R = (R << 1) & 0xFFFF
        if fb == 1:
            R ^= 0x1021

    # After running through K bits, we flush the CRC registers with 16 0's
    for _ in range(16):
        msb = (R >> 15) & 1
        R = (R << 1) & 0xFFFF
        if fb == 1:
            R ^= 0x1021
    
    # Convert rx CRC value to a binary value to compare against the calculated one
    rxCRC = 0
    for b in crcBits:
        rxCRC = (rxCRC << 1) | b
    
    # Determine if the match, return the result packed with the decoded bit array
    crcValid = (rxCRC == R)

    return decodedBits, crcValid
    
def rm3014Encoder(inputData:List[int]) -> List[int]:
    '''
    Encodes 14 input bits into 30 shorted (30,14) Reed Muller encoded output bits
    
    :param inputData: array of 1 and 0's
    '''
    # encoding is very simple, input vector (1,14) XORs with (14,30) generator matrix 'RM_30_14_GENERATOR' to yield (1,14) length output vector
    # to speed up calculatations instead of bit arrays we just use binary values
    codeWordOutput = 0
    for i, bit in enumerate(inputData):
        if bit & 1:
            codeWordOutput ^= RM_30_14_GENERATOR[i]

    # convert binary output to bit array
    outputData = [(codeWordOutput >> (29-i)) & 1 for i in range(30)]
    return outputData

def _buildRMCodebook():
    '''
    Initializes a codebook containing all possible 2^14 codewords for decoding (30,14) RM encoded data.
    '''
    # 1. Generate 2^14 gray code input possibilities

    # gray code generation inner function is taken from avanitrachhadiya2155
    def _generateGray(n):
        if n <= 0:
            return ["0"]
        if n == 1:
            return ["0","1"]
        recAns = _generateGray(n - 1)
        mainAns = []
        # Append 0 to the first half
        for i in range(len(recAns)):
            s = recAns[i]
            mainAns.append("0" + s)
        # Append 1 to the second half
        for i in range(len(recAns) - 1, -1, -1):
            s = recAns[i]
            mainAns.append("1" + s)

        return mainAns
    
    grayInput = _generateGray(14)
    grayBinaryValues = zeros(2**14,dtype=uint32)

    # converting the array form into binary, wasteful to do it at this point but idc
    for i, inputWord in enumerate(grayInput):
        u = int(inputWord, 2)
        grayBinaryValues[i] = u    
    
    # 2. Calculate the resulting codewords, note that due to gray code input, only one bit changes at a time, 
    #    this greatly simplifies the calculation and also makes searching the code book very basic.

    # first input and output codeword is all 0's
    C = zeros(2**14,dtype=uint32)
    prev_g = grayBinaryValues[0]
    codeWord = 0
    C[prev_g] = codeWord

    for i in range(1, 2**14):
        g = grayBinaryValues[i]
        # because the RM(30,14) encoding is linear, can rapidly calculate next codeword if we know the difference 
        # between the previous input and the current word
        delta = int(g ^ prev_g)
        # determine which bit changed -> this coorelates to the row in the generator matrix
        bitPosition = delta.bit_length() - 1
        # only need to perform one calculation knowing the applicable row of the generator matrix to use
        codeWord ^= uint32(RM_30_14_GENERATOR[bitPosition])

        C[g] = codeWord
        prev_g = g
    
    return C

def rm3014Decoder(inputData:List[int]) -> List[int]:
    '''
    Decodes 30 (30,14) Reed Muller encoded input bits into 14 output bits,
    performing hard minimum distance decoding using an enumerated codebook.
    
    :param inputData: array of 1 and 0's
    '''
    # There are papers on majority logic algorithm for shortned RM codes for TETRA, but I don't have access
    # and because of the nature of the usage of the RM(30,14) on access channels without any additional 
    # convolutional encoding and only being 14 bits, it's more sure-fire to just use hard table decoding 
    # instead of attempting syndrome decoding of some sort, tba perhaps

    global RM_30_14_codebook
    # if the codebook is not already populated, populate the codebook
    if RM_30_14_codebook is None:
        RM_30_14_codebook = _buildRMCodebook()
    
    #1. Convert input bit array into a 30 bit binary int value
    inputWord = 0
    for k in range(len(inputData)):
        inputWord |= (inputData[k] & 1) << (len(inputData) -1 - k)
    
    inputWord = uint32(inputWord)
    codebookLength =  1 << 14

    #2. Iterate over the generated codebook, calculating hamming distance to determine optimal input word 

    minMetric = 31
    minWord = 0

    # Note because of construction of gray code input codebook, the input value is equal to the iteration of the codebook index
    for u in range(codebookLength):
        d = (int(inputWord) ^ int(RM_30_14_codebook[u])).bit_count()
        if d < minMetric:
            minMetric = d
            minWord = u
            if minMetric == 0:
                # if we have a distance of 0, this is optimal so quit now
                break
    
    decodedBits = [(int(minWord) >> i) & 1 for i in range(14)]

    return decodedBits


###################################################################################################
# Convolutional encoding/decoding (type 2 - to - type 3 bits)

def _fetchRCPCParameters(K2:int, K3:int) -> Tuple[List[int],int,int]:
    '''
    Returns the corresponding puncturing vector, t-value, and division ratio for a given TETRA RCPC (K2,K3) ratio

    :param K2: input rate ratio numerator
    :param K3: output rate ratio denominator
    '''
    P = []
    t = 0
    divRatio = 0
    match (K2,K3):
        case(2,3):
            P = [0, 1, 4]
            t = 3
        case(1,3):
            P = [0, 1, 2, 4, 5, 6]
            t = 6
        case(292,432):
            P = [0, 1, 4]
            t = 3
            divRatio = 65
        case(148,432):
            P = [0, 1, 2, 4, 5, 6]
            t = 6
            divRatio = 35

    return P, t, divRatio

def rcpcEncoder(inputData:List[int], K2:int, K3:int) -> List[int]:
    '''
    Performs 16-state rate-compatible punctured convoltion (RCPC) encoding of rate (K2/K3) on the input data
    
    :param inputData: array of 1 and 0's
    :param K2: input rate ratio numerator
    :param K3: output rate ratio denominator
    '''

    # determine if combination of K2 and K3 are numerical valid given the length of the input data stream
    if (len(inputData) * K3) % K2 != 0:
        raise ValueError(f"RCPC encoder value of {K2}/{K3} is not valid with data length {len(inputData)}")
    
    # Step 1: encoding with 16-state mother code of rate 1/4
    encodedDataTemp = []

    # initialize the shift register as all 0's, order of register is d1, d2, d3, d4
    shiftRegister = deque([0 for _ in range(4)], maxlen=4)

    for k in range(len(inputData)):
        # add input to shift register for easy XOR'ing with generator polynominal
        input = [inputData[k]] + list(shiftRegister)
        nibbleTemp  = [0, 0, 0, 0] # order c1, c2, c3, c4
        for i in range(4):
            # for i e [0,4] calculate the c value, recall for every 1 bit input we output 5 bits
            c = 0
            for j in range(5):
                # to generate output data simple XOR operation using generator polynomial as a bit mask
                c ^= (input[j] & RCPC_MOTHER_GENERATOR[i][j])
            nibbleTemp[i] = c

        # add the 4 new bits
        encodedDataTemp.extend(nibbleTemp)
        # add new data point to top of shift reg, discard oldest bit on rightside
        shiftRegister.appendleft(inputData[k])

    # Step 2: puncturing of the mother code to obtain a 16-state RCPC code fo rate K2/K3
    #         with possible rates of: 2/3, 1/3, 292/432, 148/432

    P, t, divRatio = _fetchRCPCParameters(K2,K3)
    puncturedData = []
    
    numOutputBits = (len(inputData) * K3) // K2
    for j in range(numOutputBits):
        # j represents output data index
        i = j if divRatio == 0 else (j + (j // divRatio))
        # i indexes the puncturing pattern 
        k = 8*(i // t) + P[i % t] # P indicing equivalent to i - t(i//t) for integers
        # k indexes the mother-code bitstream

        puncturedData.append(encodedDataTemp[k])
    
    return puncturedData

def _buildRCPCTransistionTable() -> dict[int,List[tuple[int,int,List[int]]]]:
    '''
    Builds a RCPC convolution state transistion table dictionary, that enables mapping and calculations of Viterbi decoding branches
    The dictionary is in the form of a 16-key dictionary of destination states, each with a value of a 2-element list of tuples: {destination state:int : [(starting state:int, input bit:int, output bits:list), (,,)]}
    '''
    tempTable = {i: [] for i in range(16)} # each key is one of the 16 states,

    states = [[(i >> 3) & 0b1, (i >> 2) & 0b1, (i >> 1) & 0b1, i & 0b1] for i in range(16)]

    # for each of the states, calculate the possible destinations using 0 or 1 as input transistion, then log the result, indexed by destination
    for stateBits in states:
        for inputBit in [0, 1]:
            input = [inputBit] + stateBits
            # add input to shift register, i.e. the state, for easy XOR'ing with generator polynominal
            outputBits = [0, 0, 0, 0] # order c1, c2, c3, c4
            for i in range(4):
                # for i e [0,4] calculate the c value, recall for every 1 bit input we output 5 bits
                c = 0
                for j in range(5):
                    # to generate output data simple XOR operation using generator polynomial as a bit mask
                    c ^= (input[j] & RCPC_MOTHER_GENERATOR[i][j])
                outputBits[i] = c

            # destination state = shift register after input
            destinationStateBits = [inputBit] + stateBits[:-1]
            # for simplicity of lookup we just store the states as integers 0-15
            currentState = sum(x << (3-i) for i, x in enumerate(stateBits))
            destinationState = sum(x << (3-i) for i, x in enumerate(destinationStateBits))
            tempTable[destinationState].append((currentState, inputBit, outputBits))

    # for key in tempTable.keys():
    #     print(f"Destination State   Input Bit  Output Bits     Source State")
    #     print(f"{key} \t\t\t {tempTable[key][0][1]} \t {tempTable[key][0][2]} \t\t {tempTable[key][0][0]}")
    #     print(f"{key} \t\t\t {tempTable[key][1][1]} \t {tempTable[key][1][2]} \t\t {tempTable[key][1][0]}")

    return tempTable


def rcpcDecoder(inputData:List[int], payloadLength:int, K2:int, K3:int) -> List[int]:
    '''
    Performs Viterbi decoding of 16-state RCPC convolutional encoded data of rate (K2/K3) into decoded bits into an array of 'payloadLength' length
    
    :param inputData: array of 1 and 0's
    :param payloadLength: int representing the number of output payload bits expected
    :param K2: input rate ratio numerator
    :param K3: output rate ratio denominator
    '''
    global RCPC_transition_table
    # a hard-decision Viterbi decoder basically, with some complication of handling punctured data
    decodedBits = []
    # if the transition table is now built already, build it
    if RCPC_transition_table == None:
        RCPC_transition_table = _buildRCPCTransistionTable()

    # Determine the t, divRatio, and puncturing index values based on K2,K3
    P, t, divRatio = _fetchRCPCParameters(K2,K3)

    # This Viterbi decoder uses 1 data bit = 1 Viterbi transition basis
    # This results in some complication because, due to fractional puncturing, some transitions are erased from the input encoded stream, which is based on a 16-state
    # However, because we know the encoding scheme, we can calculate just in-time which bits of the mothercode we must consider, some 4-bit pairs we can just ignore
    
    # inner function for calculating output data bit index, k, and i values for puncturing 
    def map_r_to_u_b(r_idx):
        i = r_idx if divRatio == 0 else r_idx + (r_idx // divRatio)
        k = 8 * (i // t) + P[i % t]
        return (k // 4), (k % 4)

    U_total = payloadLength

    # Path metrics
    pathMetric = full(16, INF, dtype=int)
    # initial decoder state is set to 0,0,0,0 just as decoder performs
    pathMetric[0] = 0

    # Use a streaming linear register method for decoding to reduce memory depth required, but requires flushing at the end of input stream
    survivorBits = [deque(maxlen=RCPC_TRACEBACK_LENGTH) for _ in range(16)]

    j = 0
    pending = {}  # u -> List[(b, rxBit)] if we ever read ahead such as when j // divRatio increments

    # Iterate over each input data payload bit
    for u in range(U_total):

        # 1. Calculate the bit puncturing values for the observation of the mother code
        obs_u = pending.pop(u, []) if u in pending else []
        # Iterate over the data length to determine for input data bit / transistion [u] what mother code bits are used [b_r]
        while j < len(inputData):
            # calculate output bit/transition index u_r and puncture index b_r
            u_r, b_r = map_r_to_u_b(j)
            # if we have a recieve bit that lies as part of next transition due to skips, store it 
            if u_r > u:
                # cache for later and stop consuming for this u
                pending.setdefault(u_r, []).append((b_r, inputData[j]))
                j += 1
                continue
            # u_r == u
            obs_u.append((b_r, inputData[j]))
            j += 1
        obs_u.sort(key=lambda x: x[0])
        
        # Note for the last 4 output data bits, we know that they are set to 0,0,0,0
        tail_enforced = (u >= U_total - RCPC_TAIL_LENGTH)
        # Temp. update metrics
        newMetric = full(16, INF, dtype=int)
        newSurvivor = [deque(maxlen=RCPC_TRACEBACK_LENGTH) for _ in range(16)]

        # for each state, determine most likely possible transistion to it
        for destState in range(16):
            bestMetric = INF
            bestOriginState = -1
            bestInputBit = 0
            # for each state determine the most likely transition by finding the minimum path length to it
            for (originState, inputBit, outputBits) in RCPC_transition_table[destState]:
                if tail_enforced and inputBit != 0:
                    continue
                metric = pathMetric[originState]
                metric += sum(rxBit ^ outputBits[b] for (b, rxBit) in obs_u)
                if metric < bestMetric:
                    bestMetric = metric
                    bestOriginState = originState
                    bestInputBit  = inputBit

            # Find most likely path to current state considered
            newMetric[destState] = bestMetric
            if bestOriginState >=0 and bestMetric < INF:
                newSurvivor[destState] = deque(survivorBits[bestOriginState], maxlen=RCPC_TRACEBACK_LENGTH)
                newSurvivor[destState].append(bestInputBit)

        # update overall pathMetric and the survivor bit streams
        pathMetric = newMetric
        survivorBits = newSurvivor

        # stream output once we have evaluated atleast RCPC_TRACEBACK_LENGTH-1 # of recieved bits
        if u >= RCPC_TRACEBACK_LENGTH - 1:
            bestState = int(argmin(pathMetric))
            # oldest bit in that survivor corresponds to (u - (L-1))
            decodedBits.append(survivorBits[bestState][0])
            for s in range(16):
                if survivorBits[s]:
                    survivorBits[s].popleft()
    
    # the last state terminates to zero, data-aided, just flush the remaining values out
    final_state = int(argmin(pathMetric))
    remaining = list(survivorBits[final_state])
    decodedBits.extend(remaining)
    
    return decodedBits

###################################################################################################
# Interleaving/deinterleaving (type 3 - to - type 4 bits)

def blockInterleaver(inputData:List[int], K:int, a:int) -> List[int]:
    '''
    Performs (K,a) block interleaving of K input data into K output data
    
    :param inputData: array of 1 and 0's
    :param k: int representing the number of input data bits
    :param a: interleaving coefficent
    '''
    # length of output interleaved data is the same as the input data: K
    
    if len(inputData) != K:
        raise ValueError(f"Length of input data: {len(inputData)} blockinterleaver does not match passed K value of: {K}")

    if gcd(a, K) != 1:
        raise ValueError (f"{K} and {a} does not have a greatest common divisor of 1, therefore cannot perform (K,a) interleaving")

    interleavedData = [0 for _ in range(K)]
    for i in range(len(inputData)):
        interleavedData[((a * i) % K)] = inputData[i]

    return interleavedData

def blockDeInterleaver(inputData:List[int], K:int, a:int) -> List[int]:
    '''
    Performs (K,a) block deinterleaving of K input data into K output data
    
    :param inputData: array of 1 and 0's
    :param k: int representing the number of input data bits
    :param a: interleaving coefficent
    '''
    if len(inputData) != K:
        raise ValueError(f"Length of input data: {len(inputData)} blockdeinterleaver does not match passed K value of: {K}")

    if gcd(a, K) != 1:
        raise ValueError (f"{K} and {a} does not have a greatest common divisor of 1, therefore cannot perform (K,a) deinterleaving")
        
    deInterleavedData = [0 for _ in range(K)]
    a_inverse = pow(a, -1, K) # calculate the inverse of a mod, only possible if coprime, already checked above

    for i in range(len(inputData)):
        deInterleavedData[((a_inverse * i) % K)] = inputData[i]

    return deInterleavedData          

def nBlockInterleaver(inputDataBlocks:List[List[int]], M:int, N:int) -> List[List[int]]:
    '''
    Performs N block interleaving of M # of input data blocks into (M+N-1) # of output data blocks
    
    :param inputDataBlocks: list of blocks which are array of 1 and 0's each length of 432 bits
    :param M: int representing the number of input blocks passed
    :param N: int representing the number of blocks to interleave: (1,4,8)
    '''
    # N may be 1, 4, or 8, interleaving M # of 432 bit long blocks over N blocks into a sequence of (M + N - 1) blocks
    if N not in [1, 4, 8]:
        raise ValueError(f"Invalid N, for N-block interleaving, passed: {N}")
    
    if M != len(inputDataBlocks):
        raise ValueError(f"Passed M of {M} does not match # of input blocks: {len(inputDataBlocks)}")

    # Step 1, diagonal interleave M blocks in (M+N-1) blocks, each with 432 bits in them
    interleavedDataBlocksTemp = [[0 for _ in range(N_BLOCK_BIT_LENGTH)] for p in range(N+M-1)]

    if len(inputDataBlocks[0]) != N_BLOCK_BIT_LENGTH:
            raise ValueError(f"Length of datablock provided to N-block interleaver is not 432 bits long")

    for m in range((N+M-1)):
        
        # block width is = N_BLOCK_BIT_LENGTH // N
        blockWidth = (N_BLOCK_BIT_LENGTH//N)
        for k in range(N_BLOCK_BIT_LENGTH):
            j = k // blockWidth
            i = k % blockWidth

            # Source block is located at (m-j)
            if 0 <= (m-j) < M:
                interleavedDataBlocksTemp[m][k] = inputDataBlocks[m-j][j+(i*N)]
            else:
                interleavedDataBlocksTemp[m][k] = 0
    
    # Step 2, block interleave
    interleavedDataBlocks = []
    for m in range((N+M-1)):
        interleavedBlock = blockInterleaver(interleavedDataBlocksTemp[m], N_BLOCK_BIT_LENGTH, N_BLOCK_BLOCK_INTERLEAVER_A_VALUE)
        interleavedDataBlocks.append(interleavedBlock)

    return interleavedDataBlocks

def nBlockDeInterleaver(inputDataBlocks:List[List[int]], M:int, N:int) -> List[List[int]]:
    '''
    Performs N block deinterleaving of (M+N-1) # of input data blocks into  # of output data blocks
    
    :param inputDataBlocks: list of blocks which are array of 1 and 0's each length of 432 bits
    :param M: int representing the number of input blocks passed
    :param N: int representing the number of blocks to deinterleave: (1,4,8)
    '''
    # N may be 1, 4, or 8, interleaving M # of 432 bit long blocks over N blocks into a sequence of (M + N - 1) blocks
    if N not in [1, 4, 8]:
        raise ValueError(f"Invalid N, for N-block interleaving, passed: {N}")
    
    if M != (len(inputDataBlocks)-N+1):
        raise ValueError(f"Passed M of {M} and N: {N} does not align for (M+N-1) = # of input blocks: {len(inputDataBlocks)}")
   
    deInterleavedDataBlocks = [[0 for _ in range(N_BLOCK_BIT_LENGTH)] for p in range(M)]
    deInterleavedDataBlocksTemp = []
    # Step 1, deinterleave the (N+M-1) blocks
    for m in range((N+M-1)):
        deInterleavedBlock = blockDeInterleaver(inputDataBlocks[m], N_BLOCK_BIT_LENGTH, N_BLOCK_BLOCK_INTERLEAVER_A_VALUE)
        deInterleavedDataBlocksTemp.append(deInterleavedBlock)

    # Step 2, reverse diagonal block interleaving transforming (N+M-1) blocks into the original M blocks
    for m in range((N+M-1)):
        if len(inputDataBlocks[m]) != N_BLOCK_BIT_LENGTH:
            raise ValueError(f"Length of datablock provided to N-block interleaver is not 432 bits long")

        # block width is = N_BLOCK_BIT_LENGTH // N
        blockWidth = (N_BLOCK_BIT_LENGTH//N)
        for k in range(N_BLOCK_BIT_LENGTH):
            j = k // blockWidth
            i = k % blockWidth

            if 0 <= (m-j) < M:
                # The diagonal interleaving is linear so we can just reverse the assignment simply
                deInterleavedDataBlocks[m-j][j+(i*N)] = deInterleavedDataBlocksTemp[m][k]

    return deInterleavedDataBlocks

###################################################################################################
# Scrambling/descrambling (type 4 - to - type 5 bits)

def scrambler(inputData:List[int], BSCH:bool=False) -> List[int]:
    '''
    Performs scrambling of input data using a predefined scrambling polynominal
    
    :param inputData: array of 1 and 0's each length of 432 bits
    :param BSCH: bool, default false, if the logical channel is BSCH, should set to true to change scrambling initilization to all 0's
    '''
    scrambledData = []
    scramblerInitCode = []

    if not BSCH:
        scramblerInitCode = [1,1] + MCC + MNC + COLOUR_CODE
    else:
        # for BSCH with scramble with zeros
        scramblerInitCode = [1,1] + [0 for i in range(30)]

    shiftRegister = deque(scramblerInitCode, maxlen=32)

    for i in range(len(inputData)):
        p_k = shiftRegister[31] # oldest data

        scrambledData.append((inputData[i] ^ p_k))
        
        feedback = 0
        for k in TETRA_32_SCRAMBLING_SEQUENCE:
            feedback ^= shiftRegister[k]
        
        shiftRegister.popleft()
        shiftRegister.append(feedback)
    return scrambledData

def descrambler(inputData:List[int], BSCH:bool=False) -> List[int]:
    '''
    Performs descrambling of input data using a predefined scrambling polynominal
    
    :param inputData: array of 1 and 0's each length of 432 bits
    :param BSCH: bool, default false, if the logical channel is BSCH, should set to true to change scrambling initilization to all 0's
    '''
    # the scrambler is self-inverse because it is just XOR with sequence, so can apply the same operation to descramble
    return scrambler(inputData, BSCH)