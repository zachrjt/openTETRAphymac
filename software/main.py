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
import src.tetraphymac.measurements as tetraMeas

np.random.seed(10)

def spectrum_db(x, Fs, n=131072):
    x = np.asarray(x)
    X = np.fft.fftshift(np.fft.fft(x, n=n))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1/Fs))
    P = (np.abs(X)**2) / (len(x)**2 * 50)
    return f, P

def bandpower(f, P, flo, fhi):
    idx = (f >= flo) & (f <= fhi)
    return P[idx].sum()

def channel_power(f, P, fc, B, window, N):
    half = B/2
    bp = bandpower(f, P, fc-half, fc+half)
    if window == "hann":
        bp /= np.mean(np.hanning(N))
    return bp

def spectrum_and_acpr(yreal, yideal, Fs, window="ones"):
    if yideal is not None:
        if window == "ones":
            windowIdeal = np.ones(len(yideal), dtype=np.float64)
        elif window == "hann":
            windowIdeal = np.hanning(len(yideal))

    if window == "ones":
        windowReal = np.ones(len(yreal), dtype=np.float64)
    elif window == "hann":
        windowReal = np.hanning(len(yreal))

    f, Pyreal = spectrum_db(yreal*windowReal, Fs)
    if yideal is not None:
        f2, Pyideal = spectrum_db(yideal*windowIdeal, Fs)

    B = ((1.35)*tetraTx.TETRA_SYMBOL_RATE)/2

    Pch0_r  =  channel_power(f, Pyreal, 0.0,  B, window, len(yreal))
    Pch25_r = 10*np.log10((channel_power(f, Pyreal, 25e3, B/2, window, len(yreal)) + 1e-32) / (Pch0_r + 1e-32))
    Pch50_r = 10*np.log10((channel_power(f, Pyreal, 50e3, B/2, window, len(yreal)) + 1e-32) / (Pch0_r + 1e-32))
    Pch75_r = 10*np.log10((channel_power(f, Pyreal, 75e3, B/2, window, len(yreal)) + 1e-32) / (Pch0_r + 1e-32))
    Pch100_250_r = 10*np.log10((channel_power(f, Pyreal, 175e3, 150E3, window, len(yreal)) + 1e-32) / (Pch0_r + 1e-32))
    Pch250_500_r = 10*np.log10((channel_power(f, Pyreal, 375e3, 250E3, window, len(yreal)) + 1e-32) / (Pch0_r + 1e-32))
    print("\nStatistics for Real Quantized Specturm")
    print(f"Signal Power (dBm): {10*np.log10(Pch0_r) +33}")
    print(f"25khz (dBc): {Pch25_r}")
    print(f"50khz (dBc): {Pch50_r}")
    print(f"75khz (dBc): {Pch75_r}")
    print(f"100-250khz (dBc): {Pch100_250_r}")
    print(f"250-500khz (dBc): {Pch250_500_r}")

    if yideal is not None:
        print("\nStatistics for Ideal Specturm")
        Pch0_i  =  channel_power(f, Pyideal, 0.0,  B, window, len(yreal))
        Pch25_i = 10*np.log10((channel_power(f2, Pyideal, 25e3, B, window, len(yreal)) + 1e-32) / (Pch0_i + 1e-32))
        Pch50_i = 10*np.log10((channel_power(f2, Pyideal, 50e3, B, window, len(yreal)) + 1e-32) / (Pch0_i + 1e-32))
        Pch75_i = 10*np.log10((channel_power(f2, Pyideal, 75e3, B, window, len(yreal)) + 1e-32) / (Pch0_i + 1e-32))
        Pch100_250_i = 10*np.log10((channel_power(f2, Pyideal, 175e3, 150E3, window, len(yreal)) + 1e-32) / (Pch0_i + 1e-32))
        Pch250_500_i = 10*np.log10((channel_power(f2, Pyideal, 375e3, 250E3, window, len(yreal)) + 1e-32) / (Pch0_i + 1e-32))

        print(f"Signal Power (dBm): {10*np.log10(Pch0_i)+33}")
        print(f"25khz (dBc): {Pch25_i}")
        print(f"50khz (dBc): {Pch50_i}")
        print(f"75khz (dBc): {Pch75_i}")
        print(f"100-250khz (dBc): {Pch100_250_i}")
        print(f"250-500khz (dBc): {Pch250_500_i}")

    plt.figure()
    plt.plot(f, (10*np.log10(Pyreal + 1e-32)+30), label="Quantized")
    if yideal is not None:
        plt.plot(f, (10*np.log10(Pyideal + 1e-32)+30), label="Float")
    plt.grid(True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBm)")
    plt.title("FFT of BB Filters with Hanning Window")
    plt.legend()
    plt.show()

def power_envelope(yreal, yideal, Fs, overlay:False):

    n = len(yreal)
    t = np.arange(n) / Fs

    env_real = 10*np.log10(((np.abs(yreal[:n])**2) / 50) + 1e-25) + 30
    if yideal is not None:
        assert len(yreal) == len(yideal)
        env_ideal = 10*np.log10(((np.abs(yideal[:n])**2) / 50) + 1e-25) + 30


    if yideal is not None:
        if not overlay:
            _, ax = plt.subplots(2,1,sharex=True)
            ax[0].plot(t, env_ideal,     label="float with ramp")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_ylabel("|y| (dBm) ideal")

            ax[1].plot(t, env_real,     label="quantized with ramp")
            ax[1].grid(True)
            ax[1].legend()
            ax[1].set_ylabel("|y| (dBm) quant")
            ax[1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.show()
        else:
            plt.figure()
            plt.plot(t, env_real, label="quantized with ramp")
            plt.plot(t, env_ideal, label="float with ramp")
            plt.grid(True)
            plt.legend()
            plt.ylabel("|y| (dBm)")
            plt.xlabel("Time (s)")
            plt.show()
    else:
        plt.figure()
        plt.plot(t, env_real, label="quantized with ramp")
        plt.grid(True)
        plt.legend()
        plt.ylabel("|y| (dBm) quant")
        plt.xlabel("Time (s)")
        plt.show()

def main():

    # Generate burst data
    tx_real = tetraTx.RealTransmitter()
    tx_ideal = tetraTx.IdealTransmitter()

    ul_tp_rf_channel = tetraPhy.PhysicalChannel(1, False, 905.1, 918.1, tetraPhy.PhyType.TRAFFIC_CHANNEL)
    ul_cp_rf_channel = tetraPhy.PhysicalChannel(4, False, 905.2, 918.2, tetraPhy.PhyType.CONTROL_CHANNEL)

    pkt_traffic_ch = tetraLch.TCH_4_8(n=4)
    pkt_traffic_ch.encode_type5_bits(pkt_traffic_ch.generate_rnd_input(4))

    ul_tp_burst = tetraPhy.NormalUplinkBurst(ul_tp_rf_channel, 1, 1, 1)
    burst_modulation_bits = ul_tp_burst.construct_burst_sequence(pkt_traffic_ch)

    rx_real = tx_real.transmit_burst(burst_modulation_bits,
                                     (ul_tp_burst.start_ramp_period, ul_tp_burst.end_ramp_period))

    rx_ideal = tx_ideal.transmit_burst(burst_modulation_bits,
                                       (ul_tp_burst.start_ramp_period, ul_tp_burst.end_ramp_period))
    Fs2 = tetraConstants.TX_BB_SAMPLING_FACTOR * tetraTx.TETRA_SYMBOL_RATE * tetraTx.TRANSMIT_SIMULATION_SAMPLING_FACTOR
    
    # Envelope comparison
    power_envelope(rx_real, rx_ideal, Fs2, True)

    # Spectra comparison
    rp_f = tetraUtil.TX_BB_SAMPLING_FACTOR*tetraTx.TRANSMIT_SIMULATION_SAMPLING_FACTOR
    sn0 = (ul_tp_burst.start_ramp_period-1)*rp_f
    snmax = len(rx_real) - ((ul_tp_burst.end_guard_bit_period-1)*rp_f)

    print("Real digital tx ACPR results:")
    print(tetraMeas.tx_acpr_measurement(rx_real.astype(np.complex64), sn0, snmax, Fs2))
    #print("Ideal digital tx ACPR results:")
    #print(tetraMeas.tx_acpr_measurement(rx_ideal.astype(np.complex64), sn0, snmax, Fs2))

    print("Real digital tx Wideband Noise results:")
    print(tetraMeas.tx_wideband_noise_measurement(rx_real.astype(np.complex64), sn0, snmax, Fs2))
    #print("Ideal digital tx Wideband Noise results:")
    #print(tetraMeas.tx_wideband_noise_measurement(rx_ideal.astype(np.complex64), sn0, snmax, Fs2))

    tetraMeas.psd_welch(rx_real, sn0, snmax, Fs2)
    #tetraMeas.psd_welch(rx_ideal, sn0, snmax, Fs2)

    
    # Demonstrate .iq file saving ability
    # data = np.vstack((rx_real.real, rx_real.imag))
    # print(rx_real.size)
    # tetraUtil.save_burst_iqfile(data, "iqData.iq", endian="little")

    # rx_real = rx_real[ul_tp_burst.start_ramp_period*rp_f:-ul_tp_burst.end_ramp_period*rp_f]
    # rx_ideal = rx_ideal[ul_tp_burst.start_ramp_period*rp_f:-ul_tp_burst.end_ramp_period*rp_f]
    # spectrum_and_acpr(rx_real, rx_ideal, Fs2, "hann")

    #########################################################################################################

    # Demonstrate subslot uplink burst usage
    # pkt_control_ch1 = tetraLch.SCH_HU()
    # pkt_control_ch1.encode_type5_bits(pkt_control_ch1.generate_rnd_input(1))

    # pkt_control_ch2 = tetraLch.SCH_HU()
    # pkt_control_ch2.encode_type5_bits(pkt_control_ch1.generate_rnd_input(1))


    # ul_cp_burst = tetraPhy.ControlUplink(ul_cp_rf_channel, 1, 1, 1)
    # sch_hu_burst1 = ul_cp_burst.construct_burst_sequence(pkt_control_ch1)
    # # ul_null_burst = tetraPhy.Null_Halfslot_Uplink_Burst(ul_cp_rf_channel, 1, 1, 1)
    # # sch_hu_burst2 = ul_null_burst.constructBurstBitSequence()
    # sch_hu_burst2 = ul_cp_burst.construct_burst_sequence(pkt_control_ch2)

    # burst_modulation_bits2 = np.stack((sch_hu_burst2, sch_hu_burst1))

    # I_real, Q_real  = tx_real.transmit_burst(burst_modulation_bits2,
    #                                         (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period),
    #                                         (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period))

    # I_ideal, Q_ideal = tx_ideal.transmit_burst(burst_modulation_bits2,
    #                                           (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period),
    #                                           (ul_cp_burst.start_ramp_period, ul_cp_burst.end_ramp_period))

    # # Demonstrate .iq file saving ability
    # data = np.vstack((I_real, Q_real))
    # tetraUtil.save_burst_iqfile(data, "iq_files\iqData.iq", endian="little")
    # i_data, q_data = tetraUtil.read_burst_iqfile("iq_files\iqData.iq", msb_aligned=True, endian="little")
    # I_real = i_data.copy()
    # Q_real = q_data.copy()

    # scale = float((1 << tetraUtil.NUMBER_OF_FRACTIONAL_BITS))

    # I_real = I_real.astype(np.float64) / scale
    # Q_real = Q_real.astype(np.float64) / scale
    # yreal = (I_real) + 1.0j*(Q_real)
    # yreal = yreal.astype(np.complex64)

    # yideal = I_ideal + 1.0j*Q_ideal
    # yideal = yideal.astype(np.complex64)

    # Fs = tetraConstants.TX_BB_SAMPLING_FACTOR * tetraTx.TETRA_SYMBOL_RATE
    # # Envelope comparison
    # power_envelope(yreal, yideal, Fs)

    # # Spectrum comparison
    # spectrum_and_acpr(yreal, yideal, Fs)

if __name__ == '__main__':
    main()
    