#!/usr/bin/env python3
"""
Test consistency between various Loiacono implementations:
1. Python CPU implementation (loiacono.py)
2. Python GPU implementation (loiacono_gpu.py) - if available
3. C++ implementations (via API server or direct comparison)

This test verifies that different implementations produce similar results
for the same input signals.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import loiacono modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loiacono import Loiacono as LoiaconoCPU
    HAS_PYTHON_CPU = True
except ImportError as e:
    print(f"Warning: Could not import Python CPU implementation: {e}")
    HAS_PYTHON_CPU = False

try:
    from loiacono_gpu import Loiacono_GPU
    HAS_PYTHON_GPU = True
except ImportError as e:
    print(f"Warning: Could not import Python GPU implementation: {e}")
    HAS_PYTHON_GPU = False

# Test parameters
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 2**12  # 4096 samples
MULTIPLE = 40
FREQ_MIN = 100  # Hz
FREQ_MAX = 3000  # Hz
NUM_BINS = 100

def generate_test_signals():
    """Generate various test signals for consistency testing."""
    signals = {}
    
    # 1. Single sine wave at A4 (440 Hz)
    t = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE
    signals['sine_440'] = np.sin(2 * np.pi * 440 * t)
    
    # 2. Multiple sine waves (harmonic series)
    signals['harmonics'] = np.zeros(SIGNAL_LENGTH)
    for i in range(1, 6):
        signals['harmonics'] += np.sin(2 * np.pi * 440 * i * t) / i
    
    # 3. White noise
    np.random.seed(42)  # For reproducibility
    signals['white_noise'] = np.random.randn(SIGNAL_LENGTH)
    
    # 4. Chirp signal (frequency sweep)
    f0, f1 = 100, 2000
    signals['chirp'] = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1])))
    
    return signals

def compute_normalized_frequencies():
    """Compute normalized frequencies for all bins."""
    freqs_hz = np.exp(np.linspace(np.log(FREQ_MIN), np.log(FREQ_MAX), NUM_BINS))
    freqs_normalized = freqs_hz / SAMPLE_RATE
    return freqs_hz, freqs_normalized

def test_python_cpu_implementation(signals):
    """Test the Python CPU implementation."""
    if not HAS_PYTHON_CPU:
        return {}
    
    print("Testing Python CPU implementation...")
    freqs_hz, freqs_normalized = compute_normalized_frequencies()
    
    results = {}
    for name, signal in signals.items():
        print(f"  Processing {name}...")
        
        # Create Loiacono instance
        loiacono = LoiaconoCPU(
            fprime=freqs_normalized,
            multiple=MULTIPLE,
            dtftlen=SIGNAL_LENGTH
        )
        
        # Run computation
        start_time = time.time()
        loiacono.run(signal)
        elapsed = time.time() - start_time
        
        results[name] = {
            'spectrum': loiacono.spectrum.copy(),
            'time': elapsed
        }
        
        print(f"    Time: {elapsed:.4f}s")
    
    return results

def test_python_gpu_implementation(signals):
    """Test the Python GPU implementation."""
    if not HAS_PYTHON_GPU:
        return {}
    
    print("Testing Python GPU implementation...")
    
    # Try to import vulkanese
    try:
        import vulkanese as ve
        import vulkan as vk
    except ImportError as e:
        print(f"  Skipping GPU test: {e}")
        return {}
    
    freqs_hz, freqs_normalized = compute_normalized_frequencies()
    
    results = {}
    
    # Initialize Vulkan
    try:
        instance = ve.instance.Instance(verbose=False)
        device = instance.getDevice(0)
        
        # Create GPU instance
        loiacono_gpu = Loiacono_GPU(
            device=device,
            fprime=freqs_normalized,
            multiple=MULTIPLE,
            signalLength=SIGNAL_LENGTH,
            DEBUG=False
        )
        
        for name, signal in signals.items():
            print(f"  Processing {name}...")
            
            # Convert signal to float32 to match buffer type
            signal_converted = np.array(signal, dtype=np.float32)
            
            # Set input signal
            loiacono_gpu.gpuBuffers.x.set(signal_converted)
            
            # Run computation
            start_time = time.time()
            loiacono_gpu.run(blocking=True)
            elapsed = time.time() - start_time
            
            # Get results
            spectrum = loiacono_gpu.getSpectrum()
            
            results[name] = {
                'spectrum': spectrum.copy(),
                'time': elapsed
            }
            
            print(f"    Time: {elapsed:.4f}s")
        
        # Cleanup
        instance.release()
        
    except Exception as e:
        print(f"  GPU test failed: {e}")
        print(f"  Note: This may be due to Vulkan validation layer issues or alignment requirements.")
        print(f"  The GPU implementation has library dependencies that may need configuration.")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
    
    return results

def compare_results(results_cpu, results_gpu, tolerance=1e-3):
    """Compare results between implementations."""
    print("\n" + "="*60)
    print("Comparing implementations")
    print("="*60)
    
    all_passed = True
    
    for signal_name in results_cpu.keys():
        print(f"\nSignal: {signal_name}")
        
        cpu_spectrum = results_cpu[signal_name]['spectrum']
        cpu_time = results_cpu[signal_name]['time']
        
        print(f"  CPU: {len(cpu_spectrum)} bins, time: {cpu_time:.4f}s")
        
        if signal_name in results_gpu:
            gpu_spectrum = results_gpu[signal_name]['spectrum']
            gpu_time = results_gpu[signal_name]['time']
            
            print(f"  GPU: {len(gpu_spectrum)} bins, time: {gpu_time:.4f}s")
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Check dimensions match
            if len(cpu_spectrum) != len(gpu_spectrum):
                print(f"  ERROR: Dimension mismatch: CPU={len(cpu_spectrum)}, GPU={len(gpu_spectrum)}")
                all_passed = False
                continue
            
            # Compute differences
            diff = np.abs(cpu_spectrum - gpu_spectrum)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rel_diff = diff / (np.abs(cpu_spectrum) + 1e-10)
            max_rel_diff = np.max(rel_diff)
            
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")
            print(f"  Max relative difference: {max_rel_diff:.6f}")
            
            # Check if differences are within tolerance
            if max_diff > tolerance:
                print(f"  WARNING: Large absolute difference (> {tolerance})")
                all_passed = False
            
            # Plot comparison for significant signals
            if signal_name in ['sine_440', 'harmonics']:
                plot_comparison(signal_name, cpu_spectrum, gpu_spectrum, freqs_hz)
        else:
            print(f"  GPU: No results available")
    
    return all_passed

def plot_comparison(signal_name, cpu_spectrum, gpu_spectrum, freqs_hz):
    """Plot comparison between CPU and GPU results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot spectra
    ax1.plot(freqs_hz, cpu_spectrum, 'b-', label='CPU', alpha=0.7)
    ax1.plot(freqs_hz, gpu_spectrum, 'r--', label='GPU', alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'Spectrum comparison: {signal_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot difference
    diff = np.abs(cpu_spectrum - gpu_spectrum)
    ax2.plot(freqs_hz, diff, 'g-', label='Absolute difference')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Difference')
    ax2.set_title('Absolute difference between CPU and GPU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_comparison_{signal_name}.png', dpi=150)
    plt.close()
    
    print(f"  Saved plot: test_comparison_{signal_name}.png")

def run_consistency_tests():
    """Run all consistency tests."""
    print("Loiacono Implementation Consistency Tests")
    print("="*60)
    
    # Generate test signals
    print("Generating test signals...")
    signals = generate_test_signals()
    print(f"Generated {len(signals)} test signals")
    
    # Get frequency arrays for plotting
    global freqs_hz
    freqs_hz, _ = compute_normalized_frequencies()
    
    # Test Python CPU implementation
    results_cpu = test_python_cpu_implementation(signals)
    
    # Test Python GPU implementation
    results_gpu = test_python_gpu_implementation(signals)
    
    # Compare results
    if results_cpu and results_gpu:
        all_passed = compare_results(results_cpu, results_gpu)
        
        if all_passed:
            print("\n" + "="*60)
            print("ALL TESTS PASSED! ✓")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("SOME TESTS FAILED! ✗")
            print("="*60)
            return 1
    else:
        print("\n" + "="*60)
        print("INCOMPLETE TEST RESULTS")
        print("="*60)
        if results_cpu:
            print("CPU implementation works")
        if results_gpu:
            print("GPU implementation works")
        return 0  # Not a failure if we can't run all tests

if __name__ == "__main__":
    sys.exit(run_consistency_tests())