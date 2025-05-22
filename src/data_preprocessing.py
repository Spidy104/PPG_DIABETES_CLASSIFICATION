import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
import re
import os
import files

# Load metadata
metadata = pd.read_csv(files.METADATA_PATH)
print(f"Loaded metadata: {len(metadata)} subjects")

# Bandpass filter
def butter_bandpass_filter(signal, fs=1000, lowcut=0.5, highcut=10, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, signal)

# Hilbert transform to extract envelope
def hilbert_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

# Gaussian filter for smoothing
def gaussian_smooth(signal, sigma=10):
    # Create Gaussian kernel
    window = gaussian(6 * sigma, sigma)  # 6*sigma ensures the kernel covers most of the distribution
    window /= window.sum()  # Normalize kernel
    return convolve1d(signal, window, mode='reflect')

# Preprocess signal
def preprocess_signal(signal, fs=1000):
    try:
        # Step 1: Bandpass filter (0.5–16 Hz)
        bandpassed = butter_bandpass_filter(signal, fs=fs, lowcut=0.5, highcut=16, order=4)
        
        # Step 2: Hilbert transform (extract envelope)
        envelope = hilbert_envelope(bandpassed)
        
        # Step 3: Gaussian smoothing (sigma=10)
        smoothed = gaussian_smooth(envelope, sigma=10)
        
        # Step 4: Normalize to [0, 1]
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        return normalized.tolist()
    except Exception as e:
        print(f"Error preprocessing signal: {e}")
        return None

def parse_signal(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # Split by whitespace (tabs, spaces, newlines) and convert to floats
        numbers = [float(num) for num in content.split() if num]
        # Convert to NumPy array
        signal_data = np.array(numbers)
        return signal_data
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

# Load PPG signals
ppg_data = []
expected_length = 2100
for subject_id in metadata['subject_ID']:
    for segment in [1, 2, 3]:
        file_name = f'{subject_id}_{segment}.txt'
        file_path = os.path.join(files.PPG_DIR, file_name)
        if os.path.exists(file_path):
            signal_data = parse_signal(file_path)
            if signal_data is None:
                continue
            # Downsample if length is significantly more than expected (e.g., 4200 → 2100)
            if len(signal_data) > expected_length * 1.5:  # Allow some buffer (e.g., 3150)
                downsample_factor = int(len(signal_data) // expected_length)
                signal_data = signal.decimate(signal_data, downsample_factor)
                print(f"Downsampled {file_name} from {len(signal_data) * downsample_factor} to {len(signal_data)} points")
            # Truncate or pad to enforce expected length of 2100
            if len(signal_data) > expected_length:
                signal_data = signal_data[:expected_length]
            elif len(signal_data) < expected_length:
                signal_data = np.pad(signal_data, (0, expected_length - len(signal_data)), mode='constant')
            # Check length after processing
            if len(signal_data) < 2000 or len(signal_data) > 2200:
                print(f"Warning: Signal {file_name} has {len(signal_data)} points after processing, expected ~{expected_length}")
                continue
            ppg_data.append({
                'subjectID_Segment': f'{subject_id}_{segment}',
                'subject_ID': subject_id,
                'segment': segment,
                'raw_signal': signal_data.tolist(),
                'signal_length': len(signal_data)
            })
        else:
            print(f"Warning: File {file_name} not found")

ppg_data = pd.DataFrame(ppg_data)
print(f"Loaded {len(ppg_data)} PPG segments")

# Preprocess signals
ppg_data['preprocessed_signal'] = ppg_data['raw_signal'].apply(
    lambda x: preprocess_signal(np.array(x)) if x is not None else None
)

# Drop rows with failed preprocessing
ppg_data = ppg_data[ppg_data['preprocessed_signal'].notnull()]
print(f"Preprocessed {len(ppg_data)} PPG segments")

# Verify signal lengths and preview
print("\nSignal length stats:")
print(ppg_data['signal_length'].describe())
print("\nSample preprocessed signal (first 5 values, subject 2, segment 1):")
sample = ppg_data[ppg_data['subjectID_Segment'] == '2_1']['preprocessed_signal']
if not sample.empty:
    print(sample.iloc[0][:5])
else:
    print("Sample 2_1 not found; showing first available:")
    print(ppg_data['preprocessed_signal'].iloc[0][:5])

# Save preprocessed data
ppg_data[['subjectID_Segment', 'subject_ID', 'segment', 'preprocessed_signal']].to_csv(
    'preprocessed_ppg.csv', index=False
)
print("Saved preprocessed signals to 'preprocessed_ppg.csv'")