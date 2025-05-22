import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import files

# Load the CSV
df = pd.read_csv(files.PREPROCESSED_PPG_DIR)

# Convert preprocessed_signal to NumPy array
df['preprocessed_signal'] = df['preprocessed_signal'].apply(lambda x: np.array(ast.literal_eval(x)))

# Visualize a few signals
for idx in range(min(3, len(df))):  # Check first 3 signals
    plt.figure(figsize=(10, 4))
    plt.plot(df['preprocessed_signal'][idx], label='Filtered Signal')
    plt.title(f"PPG Signal {df['subjectID_Segment'][idx]}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()