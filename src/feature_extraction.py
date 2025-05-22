import pandas as pd
import numpy as np
import ast
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import files

# 1) Load & reconstruct raw PPG segments
df = pd.read_csv(files.PREPROCESSED_PPG_DIR)
df['preprocessed_signal'] = df['preprocessed_signal'].apply(lambda x: np.array(ast.literal_eval(x)))
df = df.sort_values(['subject_ID', 'segment'])

# 2) Concatenate segments per subject
combined_signals = (
    df
    .groupby('subject_ID')
    .agg({
        'preprocessed_signal': lambda segs: np.concatenate(segs.tolist()),
        'segment': 'count'
    })
    .rename(columns={'preprocessed_signal': 'combined_signal',
                     'segment': 'num_segments'})
    .reset_index()
)

# --- Feature extraction for Bagging Tree ---
def extract_broad_features(signal, fs=1000):
    peaks, _ = find_peaks(signal, distance=fs//2)
    if len(peaks) < 2:
        return None

    # time‑interval features
    intervals = np.diff(peaks) / fs
    mean_interval = intervals.mean()
    std_interval  = intervals.std()
    rmssd         = np.sqrt(np.mean(np.diff(intervals)**2))

    # amplitude & slope features
    troughs, _ = find_peaks(-signal, distance=fs//2)
    amps, slopes = [], []
    for pk, tr in zip(peaks, troughs):
        if pk < tr:
            amps.append(signal[pk] - signal[tr])
            slopes.append((signal[pk] - signal[tr]) / (tr - pk))
    mean_amplitude = np.mean(amps)   if amps   else 0
    std_amplitude  = np.std(amps)    if amps   else 0
    mean_slope     = np.mean(slopes) if slopes else 0

    # global signal stats
    mean_signal = signal.mean()
    std_signal  = signal.std()
    skewness    = pd.Series(signal).skew()
    kurtosis    = pd.Series(signal).kurtosis()

    # spectral features
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 1024))
    freq_power_05_5  = psd[(freqs>=0.5)&(freqs<=5)].sum()
    freq_power_5_10  = psd[(freqs>=5)&(freqs<=10)].sum()
    spectral_entropy = entropy(psd/psd.sum())

    return {
        'mean_interval': mean_interval,
        'std_interval':  std_interval,
        'rmssd':         rmssd,
        'mean_amplitude':mean_amplitude,
        'std_amplitude': std_amplitude,
        'mean_slope':    mean_slope,
        'mean_signal':   mean_signal,
        'std_signal':    std_signal,
        'skewness':      skewness,
        'kurtosis':      kurtosis,
        'freq_power_05_5': freq_power_05_5,
        'freq_power_5_10': freq_power_5_10,
        'spectral_entropy': spectral_entropy
    }

bagging_df = combined_signals.copy()
bagging_df['features'] = bagging_df['combined_signal'].apply(extract_broad_features)
bagging_df = bagging_df.dropna(subset=['features'])

# expand feature dict into columns
features_df = pd.DataFrame(bagging_df['features'].tolist())
bagging_df = pd.concat([bagging_df[['subject_ID', 'num_segments']], features_df], axis=1)

# select top features via RandomForest
feature_cols = features_df.columns.tolist()
X = StandardScaler().fit_transform(bagging_df[feature_cols])
y = bagging_df['mean_amplitude'].values

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

top5 = importances['feature'].head(5).tolist()
print("Top 5 Bagging‐Tree Features:", top5)

# --- SAVE: Bagging‐Tree features CSV (no combined_signal) ---
bagging_final = bagging_df[['subject_ID', 'num_segments'] + top5]
bagging_final.to_csv(
    "datasets/ppg_bagging_tree_features.csv",
    index=False
)
print("Saved Bagging‐Tree features.")

# --- Feature extraction for Specific Set ---
def extract_specific_features(signal, fs=1000):
    peaks, _ = find_peaks(signal, distance=fs//2)
    if len(peaks)<2:
        return None

    intervals = np.diff(peaks)/fs
    mean_ibi = intervals.mean()
    sdnn     = intervals.std()
    rmssd    = np.sqrt(np.mean(np.diff(intervals)**2))

    length_to_max = len(signal)/np.max(np.abs(signal)) if np.max(np.abs(signal))>0 else 0

    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 1024))
    spec_ent   = entropy(psd/psd.sum())

    return {
        'mean_ibi': mean_ibi,
        'sdnn':     sdnn,
        'rmssd':    rmssd,
        'length_to_max_ratio': length_to_max,
        'spectral_entropy':    spec_ent
    }

specific_df = combined_signals.copy()
specific_df['features'] = specific_df['combined_signal'].apply(extract_specific_features)
specific_df = specific_df.dropna(subset=['features'])
spec_feats = pd.DataFrame(specific_df['features'].tolist())
specific_df = pd.concat([specific_df[['subject_ID','num_segments']], spec_feats], axis=1)

# --- SAVE: Specific‐features CSV (no combined_signal) ---
specific_df.to_csv(
    "datasets/ppg_specific_features.csv",
    index=False
)
print("Saved Specific features.")

# (optional) visualize
plt.scatter(specific_df['mean_ibi'], specific_df['spectral_entropy'], alpha=0.6)
plt.xlabel("Mean IBI (s)"); plt.ylabel("Spectral Entropy")
plt.title("PPG Specific Features"); plt.show()
