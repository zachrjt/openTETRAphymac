# Overview
This directory contains the python TETRA code module and associated tests for simulating TETRA V1.0 (per EN 300 393-2 2.4.2) MAC and physical layers.


# Current State
The current implementation implements the following in the tetraphymac module:
1. Modulation, CRC, encoding, interleaving, and scrambling and their respective inverses
2. All TETRA V1.0 V+D logical channels, ready to accept input data from C-Plane or U-Plane control of MAC, with methods to encode, scramble, etc, and vice versa
3. Physical channel class implementation for building traffic, control, or linearization bursts using logical channels, with validation of burst construction at the timeslot level, with frame, multiframe and slot number inputs for verification
4. Transmit burst generation include non-quantized and quantized with impairments transmitter with function to export and read .iq files for PlutoSDR

# In Progress Work
1. Reciever simulation, modeling syncrhonization and decoding
2. Demodulator interface with AWGN to simulate BER and MER

# Further Goals
1. Adding iq gain, offset, and phase imbalance calibration loop to simulator as present in hw design and alreadying included in tx impairments
2. Channel impairments and fading simulators per the TETRA standard
3. Upper MAC implementation and LCC, since burst generation at the timeslot level is complete, next is handling higher level MAC implementation to build real frames