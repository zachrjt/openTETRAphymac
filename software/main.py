# ZT - 2026
import src.tetraphymac.logical_channels as lc

def main():

    tx_v_d_channel = lc.TCH_4_8(N=1)
    tx_v_d_channel.encodeType5Bits([tx_v_d_channel.generateRndInput() for _ in range(2)])

    rx_v_d_channel = lc.TCH_4_8(N=1)
    rx_v_d_channel.decodeType5Bits(tx_v_d_channel.type5Blocks)
    assert rx_v_d_channel.type1Blocks == tx_v_d_channel.type1Blocks

    
if __name__ == '__main__':
    main()