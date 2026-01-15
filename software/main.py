# ZT - 2026
import src.tetraphymac.logical_channels as lc
import src.tetraphymac.coding_scrambling as cs
import numpy as np

np.random.seed(10)

def main():
    
    # inputData = np.random.randint(0,2,size=30,dtype=np.uint8)
    # outputCRC = cs.crc16Encoder(inputData)
    # decodeCRC, valid = cs.crc16Decoder(outputCRC)
    # assert valid
    # assert (decodeCRC == inputData).all()

    # inputData = np.random.randint(0,2,size=14,dtype=np.uint8)
    # outputRM = cs.rm3014Encoder(inputData)
    # decodedRM = cs.rm3014Decoder(outputRM)
    # assert (inputData == decodedRM).all()


    # inputData = np.random.randint(0,2,size=432,dtype=np.uint8)
    # scrambled = cs.scrambler(inputData)
    # descrambled = cs.descrambler(scrambled)

    # assert (inputData == descrambled).all()

    # inputData = np.random.randint(0,2,size=432,dtype=np.uint8)
    # interleaved = cs.blockInterleaver(inputData, 103)
    # deinterleaved = cs.blockDeInterleaver(interleaved, 103)

    # assert (inputData == deinterleaved).all()

    # inputData = np.random.randint(0,2,size=(4,432),dtype=np.uint8)
    # interleaved = cs.nBlockInterleaver(inputData, 4)
    # deinterleaved = cs.nBlockDeInterleaver(interleaved, 4, 4)
    # assert (inputData == deinterleaved).all()

    tx_v_d_channel = lc.TCH_4_8(N=1)
    tx_v_d_channel.encodeType5Bits(tx_v_d_channel.generateRndInput(1))

    rx_v_d_channel = lc.TCH_4_8(N=1)
    rx_v_d_channel.decodeType5Bits(tx_v_d_channel.type5Blocks)
    assert (rx_v_d_channel.type1Blocks == tx_v_d_channel.type1Blocks).all()

    
if __name__ == '__main__':
    main()