# type: ignore
# pylint: skip-file
# flake8: noqa
import sys
import pytest
from pathlib import Path
module_path = Path(__file__).resolve().parent.parent
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
import numpy as np
import scipy as sp
import src.tetraphymac.constants as tetraConstants
import src.tetraphymac.ch_simulator as tetraCh
import matplotlib.pyplot as plt
fs = 11_520_000

def get_custom_tap():
    custom_tap = tetraConstants.PropagationTapParameters(13.7E-6, 1.0, "STATIC")
    test_tap1 = tetraCh.PropagationTap(fs, 0.0, custom_tap, None)
    return test_tap1

def get_test_signal(n: int = 2000):
    f_c = 11.520E3
    t = np.arange(0, 2000) * (1/fs)
    phase = (np.pi)/4
    test_signal = 1 + (np.sqrt(1/10))*np.cos(1*np.pi*f_c * t).astype(np.complex128)

    def window_function(n):
        sps = 1
        k = np.arange((n-2)*sps, dtype=np.float64)
        profile = 0.5 * (1.0 - np.cos(np.pi * k / (((n-2)*sps)-1)))
        lut = profile.astype(np.float64)
        lut[0] = 0
        lut[-1] = 1
        return lut

    window = window_function(202)
    test_signal[:200] = test_signal[:200] * window
    test_signal[-200:] = test_signal[-200:] * (1.0 - window)

    return test_signal

def test_same_segmentation():
    # 1. Test block invariance: tap.process(a:b) == concatenate((tap.process(a), tap.process(b)))
    # For all 3 types of bursts, should not matter
    test_signal = get_test_signal()
    

    # Start vs Start-end
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "START", 'same', False)
    blk1 = test_signal.copy()[:1000]
    blk2 = test_signal.copy()[1000:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "END", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)

    # Start vs Start-middle
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "START", 'same', False)
    blk1 = test_signal.copy()[:1000]
    blk2 = test_signal.copy()[1000:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)

    # Start vs Start-middle-end
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "START", 'same', False)
    blk1 = test_signal.copy()[:750]
    blk2 = test_signal.copy()[750:1250]
    blk3 = test_signal.copy()[1250:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "END", 'same', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(out_ref, out)

    # Start vs Start-middle-middle
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "START", 'same', False)
    blk1 = test_signal.copy()[:750]
    blk2 = test_signal.copy()[750:1250]
    blk3 = test_signal.copy()[1250:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(out_ref, out)

    # Isolated vs start-end
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:1000]
    blk2 = test_signal.copy()[1000:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "END", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)

    # Isolated vs start-middle
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:1000]
    blk2 = test_signal.copy()[1000:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)

    # Isolated vs Start-middle-end
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:750]
    blk2 = test_signal.copy()[750:1250]
    blk3 = test_signal.copy()[1250:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "END", 'same', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(out_ref, out)

    # Isolated vs Start-middle-middle
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:750]
    blk2 = test_signal.copy()[750:1250]
    blk3 = test_signal.copy()[1250:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(out_ref, out)    
    

    # Minimal split start split test
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)

    # Minimal split end split test
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:-test_tap1.required_history]
    blk2 = test_signal.copy()[-test_tap1.required_history:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)
    
    # Minimal split middle split test
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    out_ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:2*test_tap1.required_history]
    blk3 = test_signal.copy()[2*test_tap1.required_history:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "MIDDLE", 'same', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(out_ref, out)
    
def test_full():
    # 2. Test 'full' mode invariance: verify full mode does not affect subsequent 'same' operation
    # i.e. tap.process(a:b, 'full') == concatenate((tap.process(a:b, 'same'), tap.null_advance(required_history))))
    # for all 3 types of burst
    test_signal = get_test_signal()

    # Test for 'full' correctness vs flushing
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    full = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    same = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    tail = test_tap2.null_advance(test_tap2.required_history)
    out = np.concatenate((same, tail))
    assert np.allclose(full, out)

    # Test for 'full' is only observational
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()
    test_tap3 = get_custom_tap()

    same = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]
    required_extra_samples = test_tap1.required_history - test_tap1.int_delay
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    discard = test_tap3.process(np.concatenate((blk1, blk2[:required_extra_samples])), blk1.size, "START", 'full', False)
    out_b = test_tap3.process(blk2, blk2.size, "END", 'same', False)

    out = np.concatenate((out_a, out_b))
    assert np.allclose(same, out)

    # Test for 'full' is only observational with full mode
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()
    test_tap3 = get_custom_tap()

    full = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]
    required_extra_samples = test_tap1.required_history - test_tap1.int_delay
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    discard = test_tap3.process(np.concatenate((blk1, blk2[:required_extra_samples])), blk1.size, "START", 'full', False)
    out_b = test_tap3.process(blk2, blk2.size, "END", 'full', False)

    out = np.concatenate((out_a, out_b))
    assert np.allclose(full, out)

    # Test for 'full' block invariances
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    full = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:2*test_tap1.required_history]
    blk3 = test_signal.copy()[2*test_tap1.required_history:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "MIDDLE", 'same', False)
    out_c = test_tap2.process(blk3, blk3.size, "END", 'full', False)
    out = np.concatenate((out_a, out_b, out_c))
    assert np.allclose(full, out)

    # Test that calling full is non-destructive for subsequent bursts
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]


    ref_1 = test_tap1.process(blk1, blk1.size, "ISOLATED", 'full', False)
    ref_2 = test_tap1.process(blk2, blk2.size, "ISOLATED", 'same', False)
    
    out_1 = test_tap2.process(blk1, blk1.size, "ISOLATED", 'same', False)
    out_2 = test_tap2.process(blk2, blk2.size, "ISOLATED", 'same', False)
    assert np.allclose(ref_2, out_2)

def test_repeatable():
    # 3. Test Repetability, verify tap.process(a, 'same'), discard tap.process(b, 'full'), tap.process(b, 'same')
    # is equivalent with tap.process(a:b, 'same')
    test_signal = get_test_signal()

    # Test that repeatable is non-destructive / observational for same mode
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    discard = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', True)
    out = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', True)
    assert np.allclose(ref, out)

    # Test that repeatable is non-destructive / observational for full mode
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    discard = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', True)
    out = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'full', True)
    assert np.allclose(ref, out)

    # Test repeatable output is Deterministic and identical to real output
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', True)
    out = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    assert np.allclose(ref, out)

    # Test that repeatable for null_advance does not impact state/non-desctructive
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'full', False)
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    discard = test_tap2.null_advance(2000, True)
    out_b = test_tap2.process(blk2, blk2.size, "END", 'full', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(ref, out)

def test_null_advance():
    # 4. Verify null_advance repeatability
    test_signal = get_test_signal()

    # Test null_advance equivalence with calling process with zeros, when zeros.size > tap1.required_history
    test_tap1 = get_custom_tap()
    test_tap2 = get_custom_tap()

    ref_1 = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    ref_2 = test_tap1.process(np.zeros(1000, dtype=np.complex128), 1000, "ISOLATED", 'same', False)
    ref = np.concatenate((ref_1, ref_2))

    out_a = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    tail = test_tap2.null_advance(1000, False)
    out = np.concatenate((out_a, tail))

    assert np.allclose(ref, out)

    # Test that resulting states are empty
    test_tap1 = get_custom_tap()

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    _ = test_tap1.null_advance(test_tap1.required_history)

    assert test_tap1.fir_startup_priming
    assert np.allclose(test_tap1.int_delay_buffer, 0.0)

def test_gain_within_tap():
    test_signal = get_test_signal()
    custom_tap = tetraConstants.PropagationTapParameters(5.0E-6, 10**(-22.3/20), 
                                                         tetraConstants.TetraTapGainProcess.CLASS_PROCESS)


    # Test block invariance with gain process
    test_signal = get_test_signal()
    test_tap1 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))
    test_tap2 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))

    out_ref = test_tap1.process(test_signal, test_signal.size, "START", 'same', False)
    blk1 = test_signal.copy()[:1000]
    blk2 = test_signal.copy()[1000:]
    out_a = test_tap2.process(blk1, blk1.size, "START", 'same', False)
    out_b = test_tap2.process(blk2, blk2.size, "END", 'same', False)
    out = np.concatenate((out_a, out_b))
    assert np.allclose(out_ref, out)


    # Test full mode invariance with gain process
    test_tap1 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))
    test_tap2 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))
    blk1 = test_signal.copy()[:test_tap1.required_history]
    blk2 = test_signal.copy()[test_tap1.required_history:]


    ref_1 = test_tap1.process(blk1, blk1.size, "ISOLATED", 'full', False)
    ref_2 = test_tap1.process(blk2, blk2.size, "ISOLATED", 'same', False)
    
    out_1 = test_tap2.process(blk1, blk1.size, "ISOLATED", 'same', False)
    out_2 = test_tap2.process(blk2, blk2.size, "ISOLATED", 'same', False)
    assert np.allclose(ref_2, out_2)

    # Test repeatable with gain process
    test_tap1 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))
    test_tap2 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))

    ref = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    discard = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', True)
    out = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', True)
    assert np.allclose(ref, out)

    # Test null_advance with gain process
    test_tap1 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))
    test_tap2 = tetraCh.PropagationTap(fs, 20.0, custom_tap, np.random.SeedSequence(12345))

    ref_1 = test_tap1.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    ref_2 = test_tap1.process(np.zeros(1000, dtype=np.complex128), 1000, "ISOLATED", 'same', False)
    ref = np.concatenate((ref_1, ref_2))

    out_a = test_tap2.process(test_signal, test_signal.size, "ISOLATED", 'same', False)
    tail = test_tap2.null_advance(1000, False)
    out = np.concatenate((out_a, tail))

    assert np.allclose(ref, out)

if __name__ == '__main__':
    test_same_segmentation()
    test_full()
    test_repeatable()
    test_null_advance()
    test_gain_within_tap()
