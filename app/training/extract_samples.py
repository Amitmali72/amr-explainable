import pickle
import numpy as np
from pathlib import Path

DATA_PATH = r"d:\amr-explainable\data\RML2016.10a_dict.pkl"
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Extract one sample per modulation and SNR combination
samples = []
metadata = []

count = 0
for (mod, snr_val), signals in data.items():
    # just take the first signal of each
    signal = signals[0]
    samples.append(signal)
    metadata.append({"mod": mod, "snr": snr_val})
    count += 1
    if count >= 100: # let's just grab 100
        break

samples = np.array(samples)
with open(r"d:\amr-explainable\deploy\sample_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
np.save(r"d:\amr-explainable\deploy\sample_signals.npy", samples)
print("Saved 100 samples.")
