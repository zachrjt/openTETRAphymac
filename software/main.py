# ZT - 2026
import src.tetraphymac.logical_channels as lc
import src.tetraphymac.physical_channels as pc
import src.tetraphymac.modulation as tetraMod
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def main():

    # below is as an example of using the low level classes to generate burst I and Q data

    pkt_traffic_ch = lc.TCH_4_8(N=4)
    pkt_traffic_ch.encodeType5Bits(pkt_traffic_ch.generateRndInput(4))

    ul_tp_rf_channel = pc.Physical_Channel(1, False, 905.1, 918.1, pc.PhyType.TRAFFIC_CHANNEL)

    ul_tp_burst = pc.Normal_Uplink_Burst(ul_tp_rf_channel, 1, 1, 1)
    burst_modulation_bits = ul_tp_burst.constructBurstBitSequence(pkt_traffic_ch)
    print(burst_modulation_bits)

    tx = tetraMod.realTransmitter()
    data = tx.transmitBurst(burst_modulation_bits, [ul_tp_burst.startGuardBitPeriod, ul_tp_burst.endGuardBitPeriod])
   

    # 3. Oversample and zero-pad (over sample by 8 times)

if __name__ == '__main__':
    main()