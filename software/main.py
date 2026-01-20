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
    
    # 1. now we must modulate
    IQ_data = tetraMod.dqpskModulator(burst_modulation_bits)
    # fig, axs = plt.subplots(2)
    # axs[0].plot(IQ_data.real, '-', label="I")
    # axs[1].plot(IQ_data.imag, '-', label="Q")
    # axs[0].set_title("I Samples")
    # axs[0].set_ylabel("I")
    # axs[1].set_ylabel("Q")
    # axs[1].set_title("Q Samples")
    # axs[0].set_xlabel("Sample #")
    # axs[1].set_xlabel("Sample #")
    # axs[0].grid(True)
    # axs[1].grid(True)
    # plt.show()
    
     # 2. Next step quantize and upsample for our purposes to 16 bit values half scale
    up_sample_data = tetraMod.oversampleData(IQ_data, 8)
    print(up_sample_data)
   

    # 3. Oversample and zero-pad (over sample by 8 times)

if __name__ == '__main__':
    main()