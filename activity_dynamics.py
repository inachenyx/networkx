# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# # Load dataset (available at https://archive.ics.ics.uci.edu/ml/datasets/Japanese+Vowels)
# # Each sample is a time-series of 12 LPC coefficients (shape: [n_frames, 12])
# # Labels are speaker IDs (0-8)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# def load_ucj_vowels():
#     from sklearn.datasets import fetch_openml
#     data = fetch_openml(name='JapaneseVowels', version=1, parser='auto')
#     X, y = data['data'], data['target'].astype(int)
#     return X, y
#
# X, y = load_ucj_vowels()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()


def load_ucj_vowels():
    """Properly load and parse the Japanese Vowels dataset"""
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/ae.train"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/ae.test"

    def parse_file(url):
        response = requests.get(url, verify=False)
        data = []
        current_sample = []
        speaker = 0

        for line in StringIO(response.text).readlines():
            line = line.strip()
            if line.startswith('#'):  # New speaker marker
                if current_sample:
                    data.append((speaker, np.array(current_sample)))
                    current_sample = []
                speaker += 1
            elif line:  # Data line
                current_sample.append([float(x) for x in line.split()])

        if current_sample:  # Add last sample
            data.append((speaker, np.array(current_sample)))

        return data

    # Load and combine data
    train_data = parse_file(train_url)
    test_data = parse_file(test_url)
    all_data = train_data + test_data

    # Separate features and labels
    X = [sample for _, sample in all_data]
    y = [speaker for speaker, _ in all_data]

    return X, y


def encode_to_pulses(X, n_pulses=20):
    """Convert to fixed-length pulse representation"""
    pulses = []
    for sample in X:
        # Normalize per feature
        sample_norm = (sample - np.mean(sample, axis=0)) / (np.std(sample, axis=0) + 1e-8)
        # Create pulses (binary spikes)
        pulse_train = (sample_norm > 0.5).astype(float)
        # Pad/truncate
        if len(pulse_train) > n_pulses:
            pulse_train = pulse_train[:n_pulses]
        else:
            pad_width = ((0, n_pulses - len(pulse_train)), (0, 0))
            pulse_train = np.pad(pulse_train, pad_width, mode='constant')
        pulses.append(pulse_train)
    return np.array(pulses)


# Main execution
try:
    # Load and prepare data
    X, y = load_ucj_vowels()
    y = LabelEncoder().fit_transform(y)

    # Convert to fixed-length pulses
    X_pulses = encode_to_pulses(X, n_pulses=20)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_pulses, y, test_size=0.2, random_state=42
    )

    print("Data successfully loaded and processed!")
    print(f"Training set shape: {X_train.shape}")  # (n_samples, 20, 12)
    print(f"Unique speakers: {np.unique(y)}")  # Should be [0 1 2 3 4 5 6 7 8]

except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure all packages are installed:")
    print("pip install numpy scikit-learn requests")