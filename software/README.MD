# Overview
This directory contains the python TETRA code module and associated tests for simulating TETRA V1.0 (per EN 300 393-2 2.4.2) MAC and physical layers.


# Current State
The current implementation implements the following in the tetraphymac module:
1. Modulation, CRC, encoding, interleaving, and scrambling and their respective inverses
2. All TETRA V1.0 V+D logical channels, ready to accept input data from C-Plane or U-Plane control of MAC, with methods to encode, scramble, etc, and vice versa
3. Physical channel class implementation for building traffic, control, or linearization bursts using logical channels, with validation of burst construction at the timeslot, frame, multiframe, slot number levels
4. Transmit burst generation include non-quantized and quantized with impairments transmitter with function to export and read .iq files for PlutoSDR
5. Channel simulation with block-based processing, featuring fractional, abitrary delay taps, LOS, Rice, and Rayleigh fading gain doppler channel simulation


# In Progress Work
1. Test Reciever for verifying transmit performance
3. Reciever simulation, modeling syncrhonization and decoding

# Further Goals
1. Adding iq gain, offset, and phase imbalance calibration loop to simulator as present in hw design and alreadying included in tx impairments
2. Upper MAC implementation and LCC, since burst generation at the timeslot level is complete, next is handling higher level MAC implementation to build real frames