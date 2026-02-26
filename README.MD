# Overview
This repository contains the hardware and software components of my open TETRA V1.0 project. The goal of this project is to implement a TETRA V1.0 V+D compliant Mobile Station (MS) from the ground up. While this is a personal project, with the aim to sharpen my VHDL, RF design, and C++ skills, my hope is that it can have some use for someone.

# Current State
While this is a hardware focused project, it is necessary to develop an in-depth python emulation of the TETRA V1.0 standard in order to support hardware development. So while hardware design and development has already started and work is being done on both hardware and simulation software, the primary short term goals are completing the simulation software to support:

1. Logical and burst layer binary data to design and verify FPGA digital blocks _(COMPLETE)_
2. RF Burst waveforms to support Rx and TX IF AGC/ALC design work, and verification _(COMPLETE)_
3. The ability to simulate the effect of RF impairments in the baseband processing design such as adjacent (within channel select filter) and co-channel interference, fading, frequency offsets, on BER and MER _(in progress)_
4. Provide a platform to verify not only lower MAC performance with IQ waveforms, but also upper MAC behaviour by acting as a BS emulator in-loop with a PlutoSDR _(this is more of a stretch goal, verifying PHY and lower MAC is main goal)_

As the software is complete in meeting #1 and #2, below is a comparison measurement of the expected and actual uplink traffic burst waveform, which was transmitted by a PlutoSDR and measured zero-span by a spectrum analyzer:

![alpha revision hardware system block diagram](./hardware/design/images/tch_4_8_ul_burst_measurement.png)

## Hardware
System level design of the Rx portion of the RF front end is underway as is the baseband portion of the Tx portion. An RF analysis of the Rx chain is 60% complete, once python simulations are ready it will be possible to complete as currently figures for determining reciever sensitivity in terms of C/N0 are not known for the various burst types. An in-progress block diagram is provided below, not all up-to date Rx or Tx detail is included at this moment.

![alpha revision hardware system block diagram](./hardware/design/hw_block_diagram.png)

### Short Term Hardware Goals
1. Begin development of FPGA digital and processing blocks _(Tx DSP and DSP slice allocation complete)_
2. Complete revisions to Rx RF design and Tx baseband to modulator design (_in progress_)
3. Design an RF test board for ALC/AGC design, testing of wideband lossy coupler, and for VNA coplanar waveguide impedance verification for PCB stackup

## Software
Software development of a python TETRA V1.0 emulator is well underway. The following has been accomplished, and is further detailed in the `/software` directory:
1. Complete Lower MAC layer implementation
2. Complete TDMA burst construction and validation, simulation of ideal and real quantized transmitters with impairments
3. Ability to save and read .iq files to replay on an SDR

### Short Term Software Goals
1. Complete baseband sampling, filtering, and power ramping to generate burst I,Q waveforms for PlutoSDR _(COMPLETE)_
2. Transmitter analog filtering, DAZ ZOH, and noise as well _(COMPLETE)_
3. Receiver synchronization, sampling, filtering and BER/MER simulations _(in progress)_
4. Work on Upper MAC layer and LLC, to utilize logical channels and fill them with actual data _(upper MAC is less of a priority than hw design)_