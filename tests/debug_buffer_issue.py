#!/usr/bin/env python3
"""
Debug the buffer size issue in Loiacono_GPU
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from loiacono_gpu import Loiacono_GPU
    import vulkanese as ve
    import vulkan as vk
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test parameters
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 2**12  # 4096 samples
MULTIPLE = 40
FREQ_MIN = 100  # Hz
FREQ_MAX = 3000  # Hz
NUM_BINS = 100

# Compute frequencies like the test does
freqs_hz = np.exp(np.linspace(np.log(FREQ_MIN), np.log(FREQ_MAX), NUM_BINS))
freqs_normalized = freqs_hz / SAMPLE_RATE

print(f"Number of frequency bins: {len(freqs_normalized)}")
print(f"freqs_normalized shape: {freqs_normalized.shape}")
print(f"freqs_normalized dtype: {freqs_normalized.dtype}")
print(f"First few values: {freqs_normalized[:5]}")

# Try to create the GPU instance
try:
    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    
    print(f"\nCreating Loiacono_GPU with {len(freqs_normalized)} frequency bins...")
    
    # Create GPU instance
    loiacono_gpu = Loiacono_GPU(
        device=device,
        fprime=freqs_normalized,
        multiple=MULTIPLE,
        signalLength=SIGNAL_LENGTH,
        DEBUG=True
    )
    
    print("Success! Loiacono_GPU created.")
    
    # Test with a simple signal
    t = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE
    test_signal = np.sin(2 * np.pi * 440 * t)
    
    print(f"\nTest signal shape: {test_signal.shape}")
    print(f"Test signal dtype: {test_signal.dtype}")
    
    # Set input signal
    loiacono_gpu.gpuBuffers.x.set(test_signal)
    print("Input signal set successfully.")
    
    # Run computation
    loiacono_gpu.run(blocking=True)
    print("Computation completed.")
    
    # Get results
    spectrum = loiacono_gpu.getSpectrum()
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"First few spectrum values: {spectrum[:5]}")
    
    # Cleanup
    instance.release()
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to debug the buffer issue
    print("\n--- Debugging buffer creation ---")
    print(f"freqs_normalized size: {freqs_normalized.size}")
    print(f"freqs_normalized itemsize: {freqs_normalized.itemsize}")
    print(f"Total bytes: {freqs_normalized.size * freqs_normalized.itemsize}")
    
    # Check if it's a float32 vs float64 issue
    if freqs_normalized.dtype != np.float32:
        print(f"\nTrying with float32 conversion...")
        freqs_normalized_f32 = freqs_normalized.astype(np.float32)
        print(f"Converted to float32: {freqs_normalized_f32.dtype}")
        
        try:
            # Try creating buffer manually to debug
            buffer = ve.buffer.StorageBuffer(
                device=device,
                name="test_f",
                memtype="float",
                qualifier="readonly",
                dimensionVals=[len(freqs_normalized_f32)],
                memProperties=0 | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            )
            print(f"Buffer created with dimensionVals: {[len(freqs_normalized_f32)]}")
            print(f"Buffer expected size: ?")
            
            # Try to set the data
            buffer.set(freqs_normalized_f32)
            print("Buffer set successfully with float32 data")
            
        except Exception as e2:
            print(f"Buffer creation/set error: {e2}")
            import traceback
            traceback.print_exc()