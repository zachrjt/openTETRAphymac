# ZT - 2026
import src.tetraphymac.logical_channels as lc
import src.tetraphymac.physical_channels as pc
import src.tetraphymac.modulation as tetraMod
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def spectrum_db(x, Fs, N=131072):
    x = np.asarray(x)
    X = np.fft.fftshift(np.fft.fft(x, n=N))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
    P = np.abs(X)**2 / 50
    return f, P

def bandpower(f, P, flo, fhi):
    idx = (f >= flo) & (f <= fhi)
    return P[idx].sum()

def channel_power(f, P, fc, B):
    half = B/2
    return bandpower(f, P, fc-half, fc+half)



def main():

    # below is as an example of using the low level classes to generate burst I and Q data

    pkt_traffic_ch = lc.TCH_4_8(N=4)
    pkt_traffic_ch.encodeType5Bits(pkt_traffic_ch.generateRndInput(4))

    ul_tp_rf_channel = pc.Physical_Channel(1, False, 905.1, 918.1, pc.PhyType.TRAFFIC_CHANNEL)

    ul_tp_burst = pc.Normal_Uplink_Burst(ul_tp_rf_channel, 1, 1, 1)
    burst_modulation_bits = ul_tp_burst.constructBurstBitSequence(pkt_traffic_ch)
    print(burst_modulation_bits)

    tx_real = tetraMod.realTransmitter()
    tx_ideal = tetraMod.idealTransmitter()

    I_real, Q_real = tx_real.transmitBurst(burst_modulation_bits, [ul_tp_burst.startGuardBitPeriod, ul_tp_burst.endGuardBitPeriod])

    I_ideal, Q_ideal = tx_ideal.transmitBurst(burst_modulation_bits, [ul_tp_burst.startGuardBitPeriod, ul_tp_burst.endGuardBitPeriod])
    
    yreal = (I_real) + 1.0j*(Q_real)
    yreal = yreal.astype(np.complex64)
    # Convert yreal to float to compare more easily
    yreal /= (2 ** tetraMod.NUMBER_OF_FRACTIONAL_BITS)

    yideal = I_ideal + 1.0j*Q_ideal
    yideal = yideal.astype(np.complex64)
    print(len(yideal))
    print(len(yreal))

    windowReal = np.hanning(len(yreal))
    windowIdeal = np.hanning(len(yideal))
    Fs = (tetraMod.BASEBAND_SAMPLING_FACTOR * tetraMod.TETRA_SYMBOL_RATE)

    f, Pyreal = spectrum_db(yreal*windowReal, Fs)
    f2, Pyideal = spectrum_db(yideal*windowIdeal, Fs)

    B = ((1.35)*tetraMod.TETRA_SYMBOL_RATE)/2

    Pch0_r  =  channel_power(f, Pyreal, 0.0,  B)
    Pch25_r = 10*np.log10((channel_power(f, Pyreal, 25e3, B/2) + 1e-30) / (Pch0_r + 1e-30))
    Pch50_r = 10*np.log10((channel_power(f, Pyreal, 50e3, B/2) + 1e-30) / (Pch0_r + 1e-30))
    Pch75_r = 10*np.log10((channel_power(f, Pyreal, 75e3, B/2) + 1e-30) / (Pch0_r + 1e-30))
    Pch100_250_r = 10*np.log10((channel_power(f, Pyreal, 175e3, 150E3) + 1e-30) / (Pch0_r + 1e-30))
    Pch250_500_r = 10*np.log10((channel_power(f, Pyreal, 375e3, 250E3) + 1e-30) / (Pch0_r + 1e-30))
    print("Statistics for Real Quantized Specturm")
    print(f"Signal Power (dBm): {10*np.log10(Pch0_r)}")
    print(f"25khz (dBc): {Pch25_r}")
    print(f"50khz (dBc): {Pch50_r}")
    print(f"75khz (dBc): {Pch75_r}")
    print(f"100-250khz (dBc): {Pch100_250_r}")
    print(f"250-500khz (dBc): {Pch250_500_r}")

    print("\nStatistics for Ideal Specturm")
    Pch0_i  =  channel_power(f, Pyideal, 0.0,  B)
    Pch25_i = 10*np.log10((channel_power(f2, Pyideal, 25e3, B) + 1e-30) / (Pch0_i + 1e-30))
    Pch50_i = 10*np.log10((channel_power(f2, Pyideal, 50e3, B) + 1e-30) / (Pch0_i + 1e-30))
    Pch75_i = 10*np.log10((channel_power(f2, Pyideal, 75e3, B) + 1e-30) / (Pch0_i + 1e-30))
    Pch100_250_i = 10*np.log10((channel_power(f2, Pyideal, 175e3, 150E3) + 1e-30) / (Pch0_i + 1e-30))
    Pch250_500_i = 10*np.log10((channel_power(f2, Pyideal, 375e3, 250E3) + 1e-30) / (Pch0_i + 1e-30))

    print(f"Signal Power (dBm): {10*np.log10(Pch0_i)}")
    print(f"25khz (dBc): {Pch25_i}")
    print(f"50khz (dBc): {Pch50_i}")
    print(f"75khz (dBc): {Pch75_i}")
    print(f"100-250khz (dBc): {Pch100_250_i}")
    print(f"250-500khz (dBc): {Pch250_500_i}")

    plt.figure()
    plt.plot(f, 20*np.log10(np.abs(Pyreal) + 1e-15), label="Quantized")
    plt.plot(f, 20*np.log10(np.abs(Pyideal) + 1e-15), label="Float")
    plt.grid(True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of BB Filters with Hanning Window")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()