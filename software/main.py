# pylint: skip-file
# flake8: noqa
# type: ignore
# ZT - 2026
import numpy as np
import matplotlib.pyplot as plt
import src.tetraphymac.logical_channels as tetraLch
import src.tetraphymac.physical_channels as tetraPhy
import src.tetraphymac.transmitter as tetraTx
import src.tetraphymac.tx_rx_utilities as tetraUtil
import src.tetraphymac.constants as tetraConstants

np.random.seed(10)

def spectrum_db(x, Fs, n=131072):
    x = np.asarray(x)
    X = np.fft.fftshift(np.fft.fft(x, n=n))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1/Fs))
    X /= len(x)
    P = np.abs(X)**2 / 50
    return f, P

def bandpower(f, P, flo, fhi):
    idx = (f >= flo) & (f <= fhi)
    return P[idx].sum()

def channel_power(f, P, fc, B):
    half = B/2
    return bandpower(f, P, fc-half, fc+half)

def spectrum_and_acpr(yreal, yideal, Fs):
    if yideal is not None:
        #windowIdeal = np.hanning(len(yideal))
        windowIdeal = np.ones(len(yideal), dtype=np.float64)

    #windowReal = np.hanning(len(yreal))
    windowReal = np.ones(len(yreal), dtype=np.float64)


    f, Pyreal = spectrum_db(yreal*windowReal, Fs)
    if yideal is not None:
        f2, Pyideal = spectrum_db(yideal*windowIdeal, Fs)

    B = ((1.35)*tetraTx.TETRA_SYMBOL_RATE)/2

    Pch0_r  =  channel_power(f, Pyreal, 0.0,  B)
    Pch25_r = 10*np.log10((channel_power(f, Pyreal, 25e3, B/2) + 1e-12) / (Pch0_r + 1e-12))
    Pch50_r = 10*np.log10((channel_power(f, Pyreal, 50e3, B/2) + 1e-12) / (Pch0_r + 1e-12))
    Pch75_r = 10*np.log10((channel_power(f, Pyreal, 75e3, B/2) + 1e-12) / (Pch0_r + 1e-12))
    Pch100_250_r = 10*np.log10((channel_power(f, Pyreal, 175e3, 150E3) + 1e-12) / (Pch0_r + 1e-12))
    Pch250_500_r = 10*np.log10((channel_power(f, Pyreal, 375e3, 250E3) + 1e-12) / (Pch0_r + 1e-12))
    print("\nStatistics for Real Quantized Specturm")
    print(f"Signal Power (dBm): {10*np.log10(Pch0_r)}")
    print(f"25khz (dBc): {Pch25_r}")
    print(f"50khz (dBc): {Pch50_r}")
    print(f"75khz (dBc): {Pch75_r}")
    print(f"100-250khz (dBc): {Pch100_250_r}")
    print(f"250-500khz (dBc): {Pch250_500_r}")

    if yideal is not None:
        print("\nStatistics for Ideal Specturm")
        Pch0_i  =  channel_power(f, Pyideal, 0.0,  B)
        Pch25_i = 10*np.log10((channel_power(f2, Pyideal, 25e3, B) + 1e-12) / (Pch0_i + 1e-12))
        Pch50_i = 10*np.log10((channel_power(f2, Pyideal, 50e3, B) + 1e-12) / (Pch0_i + 1e-12))
        Pch75_i = 10*np.log10((channel_power(f2, Pyideal, 75e3, B) + 1e-12) / (Pch0_i + 1e-12))
        Pch100_250_i = 10*np.log10((channel_power(f2, Pyideal, 175e3, 150E3) + 1e-12) / (Pch0_i + 1e-12))
        Pch250_500_i = 10*np.log10((channel_power(f2, Pyideal, 375e3, 250E3) + 1e-12) / (Pch0_i + 1e-12))

        print(f"Signal Power (dBm): {10*np.log10(Pch0_i)}")
        print(f"25khz (dBc): {Pch25_i}")
        print(f"50khz (dBc): {Pch50_i}")
        print(f"75khz (dBc): {Pch75_i}")
        print(f"100-250khz (dBc): {Pch100_250_i}")
        print(f"250-500khz (dBc): {Pch250_500_i}")

    plt.figure()
    plt.plot(f, 10*np.log10(Pyreal + 1e-12), label="Quantized")
    if yideal is not None:
        plt.plot(f, 10*np.log10(Pyideal + 1e-12), label="Float")
    plt.grid(True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBm)")
    plt.title("FFT of BB Filters with Hanning Window")
    plt.legend()
    plt.show()

def power_envelope(yreal, yideal, Fs):

    n = len(yreal)
    t = np.arange(n) / Fs

    env_real = 20*np.log10((np.abs(yreal[:n]) / 50) + 1e-12)
    if yideal is not None:
        assert len(yreal) == len(yideal)
        env_ideal = 20*np.log10((np.abs(yideal[:n]) / 50) + 1e-12)


    if yideal is not None:
        _, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(t, env_ideal,     label="float with ramp")
        ax[0].grid(True)
        ax[0].legend()
        ax[0].set_ylabel("|y| (dB) ideal")

        ax[1].plot(t, env_real,     label="quantized with ramp")
        ax[1].grid(True)
        ax[1].legend()
        ax[1].set_ylabel("|y| (dB) quant")
        ax[1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.plot(t, env_real, label="quantized with ramp")
        plt.grid(True)
        plt.legend()
        plt.ylabel("|y| (dB) quant")
        plt.xlabel("Time (s)")
        plt.show()

def main():

    tx_real = tetraTx.RealTransmitter()
    tx_ideal = tetraTx.IdealTransmitter()

    ul_tp_rf_channel = tetraPhy.PhysicalChannel(1, False, 905.1, 918.1, tetraPhy.PhyType.TRAFFIC_CHANNEL)
    ul_cp_rf_channel = tetraPhy.PhysicalChannel(4, False, 905.2, 918.2, tetraPhy.PhyType.CONTROL_CHANNEL)

    # # below is as an example of using the low level classes to generate burst I and Q data
    # # Then using it generate 7 bursts of continous uplink data

    # # Encode the bursts in (L = 4 + n-1 = 7) bursts
    pkt_traffic_ch = tetraLch.TCH_4_8(n=4)
    pkt_traffic_ch.encode_type5_bits(pkt_traffic_ch.generate_rnd_input(4))

    print(pkt_traffic_ch.type_5_blocks.shape)

    #Generate physical channel burst wrapper:
    ul_tp_burst = tetraPhy.NormalUplinkBurst(ul_tp_rf_channel, 1, 1, 1)

    burst_modulation_bits = np.empty(shape=(7, 510), dtype=np.uint8)
    ramp_data = []
    for n in range(7):
        ramp_tuple = (False, False)
        if n == 0:
            # at start of bursts we ramp up
            ramp_tuple = (True, False)
        elif n==6:
            # at end of bursts we ramp down
            ramp_tuple = (False, True)
        else:
            # inner bursts we are continous no ramping
            ramp_tuple = (False, False)

        print(f"Ramp State for {n}: {ramp_tuple}")
        burst_modulation_bits[n][:] =  ul_tp_burst.construct_burst_sequence(pkt_traffic_ch,
                                                                            ramp_up_down_state=ramp_tuple)
        # Drop first burst bits
        pkt_traffic_ch.type_5_blocks = pkt_traffic_ch.type_5_blocks[1:]
        ramp_data.append((ul_tp_burst.start_ramp_period, ul_tp_burst.end_ramp_period))
        print(f"Ramp offsets for {n}: {ramp_data[n]}")

    offsetF = tetraConstants.TX_BB_SAMPLING_FACTOR*255
    I_data = np.zeros(shape=(offsetF*7), dtype=np.int64)
    Q_data = np.zeros(shape=(offsetF*7), dtype=np.int64)

    for n in range(7):
        print(ramp_data[n])
        s_offset = (n)*offsetF
        e_offset = (n+1)*offsetF
        I_data[s_offset:e_offset],  Q_data[s_offset:e_offset] = tx_real.transmit_burst(burst_modulation_bits[n],
                                                                                       ramp_data[n])

    data = np.vstack((I_data, Q_data))
    tetraUtil.save_burst_iqfile(data, "iqDataCont.iq", endian="little")
    i_data, q_data = tetraUtil.read_burst_iqfile("iqDataCont.iq", msb_aligned=True, endian="little")
    I_data = i_data.copy()
    Q_data = q_data.copy()

    scale = float((1 << tetraUtil.NUMBER_OF_FRACTIONAL_BITS))

    I_real = I_data.astype(np.float64) / scale
    Q_real = Q_data.astype(np.float64) / scale
    yreal = (I_real) + 1.0j*(Q_real)
    yreal = yreal.astype(np.complex64)

    Fs = tetraConstants.TX_BB_SAMPLING_FACTOR * tetraTx.TETRA_SYMBOL_RATE

    #Envelope comparison
    power_envelope(yreal, None, Fs)

    #Spectra comparison
    spectrum_and_acpr(yreal, None, Fs)

    pkt_traffic_ch = tetraLch.TCH_4_8(n=4)
    pkt_traffic_ch.encode_type5_bits(pkt_traffic_ch.generate_rnd_input(4))

    ul_tp_rf_channel = tetraPhy.PhysicalChannel(1, False, 905.1, 918.1, tetraPhy.PhyType.TRAFFIC_CHANNEL)

    ul_tp_burst = tetraPhy.NormalUplinkBurst(ul_tp_rf_channel, 1, 1, 1)
    burst_modulation_bits = ul_tp_burst.construct_burst_sequence(pkt_traffic_ch)


    I_real, Q_real  = tx_real.transmit_burst(burst_modulation_bits,
                                            (ul_tp_burst.start_ramp_period, ul_tp_burst.end_ramp_period))

    I_ideal, Q_ideal = tx_ideal.transmit_burst(burst_modulation_bits,
                                              (ul_tp_burst.start_ramp_period, ul_tp_burst.end_ramp_period))

    #Demonstrate .iq file saving ability
    # data = np.vstack((I_real, Q_real))
    # tetraUtil.save_burst_iqfile(data, "iqData.iq", endian="little")
    # i_data, q_data = tetraUtil.read_burst_iqfile("iqData.iq", msb_aligned=True, endian="little")
    # I_real = i_data.copy()
    # Q_real = q_data.copy()

    scale = float((1 << tetraUtil.NUMBER_OF_FRACTIONAL_BITS))

    I_real = I_real.astype(np.float64) / scale
    Q_real = Q_real.astype(np.float64) / scale
    yreal = (I_real) + 1.0j*(Q_real)
    yreal = yreal.astype(np.complex64)

    yideal = I_ideal + 1.0j*Q_ideal
    yideal = yideal.astype(np.complex64)

    Fs = tetraConstants.TX_BB_SAMPLING_FACTOR * tetraTx.TETRA_SYMBOL_RATE

    # Envelope comparison
    power_envelope(yreal, yideal, Fs)

    # Spectra comparison
    spectrum_and_acpr(yreal, yideal, Fs)

    #########################################################################################################

    # Demonstrate subslot uplink burst usage
    pkt_control_ch1 = tetraLch.SCH_HU()
    pkt_control_ch1.encode_type5_bits(pkt_control_ch1.generate_rnd_input(1))

    pkt_control_ch2 = tetraLch.SCH_HU()
    pkt_control_ch2.encode_type5_bits(pkt_control_ch1.generate_rnd_input(1))


    ul_cp_burst = tetraPhy.ControlUplink(ul_cp_rf_channel, 1, 1, 1)
    sch_hu_burst1 = ul_cp_burst.construct_burst_sequence(pkt_control_ch1)
    # ul_null_burst = tetraPhy.Null_Halfslot_Uplink_Burst(ul_cp_rf_channel, 1, 1, 1)
    # sch_hu_burst2 = ul_null_burst.constructBurstBitSequence()
    sch_hu_burst2 = ul_cp_burst.construct_burst_sequence(pkt_control_ch2)

    burst_modulation_bits2 = np.stack((sch_hu_burst2, sch_hu_burst1))

    I_real, Q_real  = tx_real.transmit_burst(burst_modulation_bits2,
                                            (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period),
                                            (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period))

    I_ideal, Q_ideal = tx_ideal.transmit_burst(burst_modulation_bits2,
                                              (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period),
                                              (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period))

    # Demonstrate .iq file saving ability
    data = np.vstack((I_real, Q_real))
    tetraUtil.save_burst_iqfile(data, "iqData.iq", endian="little")
    i_data, q_data = tetraUtil.read_burst_iqfile("iqData.iq", msb_aligned=True, endian="little")
    I_real = i_data.copy()
    Q_real = q_data.copy()

    scale = float((1 << tetraUtil.NUMBER_OF_FRACTIONAL_BITS))

    I_real = I_real.astype(np.float64) / scale
    Q_real = Q_real.astype(np.float64) / scale
    yreal = (I_real) + 1.0j*(Q_real)
    yreal = yreal.astype(np.complex64)

    yideal = I_ideal + 1.0j*Q_ideal
    yideal = yideal.astype(np.complex64)

    Fs = tetraConstants.TX_BB_SAMPLING_FACTOR * tetraTx.TETRA_SYMBOL_RATE
    # Envelope comparison
    power_envelope(yreal, yideal, Fs)

    # Spectrum comparison
    spectrum_and_acpr(yreal, yideal, Fs)

if __name__ == '__main__':
    main()
    