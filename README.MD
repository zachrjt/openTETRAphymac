This directory contains the python TETRA code module and associated tests for simulating TETRA V1.0 (per EN 300 393-2 2.4.2) MAC and physical layers.

Currently implements:
1. Modulation, CRC, encoding, interleaving, and scrambling and their respective inverses
2. All V+D logical channels, ready to accept input data from C-Plane or U-Plane control of MAC, with methods to encode, scramble, etc, and vice versa
3. Physical channel implementation for building traffic, control, or linearization bursts using logical channels, with validation burst construction at the timeslot level

In progress:
1. Refactoring logical channels and physical layer to use numpy and preallocation to improve speeds

Future goals
1. Baseband burst IQ data generation, with oversampling, RRC, and power ramping
2. Demodulator interface with AWGN to simulate BER and MER
3. LLC and DLL of MAC, since burst generation at the timeslot is complete next is handling higher level MAC implementation for U-Plane and C-Plane
4. Channel impairments and fading for further BER and MER simulation, with output to Pluto-SDR compatible format for IQ replay 
5. Reciever demodulator synchronization to simulate synchronization methods