import sys
import random
import time
import importlib.resources as pkg_resources

from springer_segmentation.duration_distributions import DataDistribution
from springer_segmentation.segmentation_model import SegmentationModel
from springer_segmentation.utils import get_wavs_and_tsvs, get_heart_rate_from_tsv, create_segmentation_array, create_train_test_split, upsample_states
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import librosa
import os


def train(data_dir, heartrates_from_tsv=False):
    # Get training recordings and segmentations
    train_recordings, train_segmentations, \
        test_recordings, test_segmentations = create_train_test_split(directory=data_dir,
                                                                      frac_train = 0.74,
                                                                      max_train_size=1000,
                                                                      max_test_size=300)
    print(len(train_recordings))

    # Preprocess into clips with annotations of the same length
    clips = []
    annotations = []
    for rec, seg in zip(train_recordings, train_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # Train the feature_prob_model
    t0 = time.time()
    print("start training")
    model = SegmentationModel()
    data_distribution = DataDistribution(train_segmentations)
    model.fit(clips, annotations, data_distribution=data_distribution)
    # Save model after training
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_model.pkl"))
    model.save(model_path)
    print("finish training after " + str(round(time.time() - t0)) + "s")

    # Process test set in to clips and annotations
    clips = []
    annotations = []
    heartrates = []
    for rec, seg in zip(test_recordings, test_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        if heartrates_from_tsv:
            heartrate = get_heart_rate_from_tsv(seg)
            for _ in range(len(clipped_recording)):
                heartrates.append(heartrate)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # Evaluate performance on clips
    idx = 0
    accuracies = np.zeros(len(clips))
    weights = np.zeros(len(clips))
    for rec, seg in zip(clips, annotations):
        if heartrates_from_tsv:
            annotation = model.predict(rec, heart_rate=heartrates[idx])
        else:
            annotation = model.predict(rec)
        print(idx)
        accuracies[idx] = (seg == annotation).mean()
        weights[idx] = seg.shape[0]
        idx += 1
    print(f"average accuracy: {accuracies.mean()}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights)}")

def calculate_segmentation(wav_path, target_sr=4000, clip_duration_sec=6):
    # Load model from package
    model_path = pkg_resources.files("springer_segmentation").joinpath("saved_model.pkl")
    model = SegmentationModel.load(model_path)

    # Load and process audio
    signal, sr = librosa.load(wav_path, sr=target_sr)
    clip_len = target_sr * clip_duration_sec
    signal = signal[:clip_len]
    if len(signal) < clip_len:
        signal = np.pad(signal, (0, clip_len - len(signal)))

    prediction = model.predict(signal)
    upsampled = upsample_states(prediction, old_fs=50, new_fs=target_sr, new_length=len(signal))

    return {
        "segmentation_50Hz": prediction,
        "segmentation_upsampled": upsampled,
        "resampled_audio": signal,
        "sampling_rate": target_sr
    }

def plot_segmentation_result(result_dict, title="Waveform with Predicted Segmentation Overlay"):
    """
    Plots the waveform and upsampled segmentation result.

    Parameters:
    - result_dict: dict returned by predict_wav_clip()
    - title: str, plot title
    """

    signal = result_dict["resampled_audio"]
    upsampled = result_dict["segmentation_upsampled"]
    sr = result_dict["sampling_rate"]
    duration_sec = len(signal) / sr

    time = np.linspace(0, duration_sec, len(signal))

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, label="Waveform", alpha=0.7)
    plt.plot(time, upsampled / 4 * np.max(signal), label="Segmentation (states)", linewidth=2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / State")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_segmentation(result_dict):
    states = result_dict["segmentation_upsampled"].astype(int)
    sr = result_dict["sampling_rate"]

    state_names = {1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
    durations = {k: [] for k in state_names.keys()}
    cycle_lengths = []

    last_s1 = None
    for i in range(1, len(states)):
        curr, prev = states[i], states[i - 1]
        if curr != prev:
            start = i
            label = curr
            end = start
            while end < len(states) and states[end] == label:
                end += 1
            durations[label].append((end - start) / sr)

            if label == 1:  # S1
                if last_s1 is not None:
                    cycle_lengths.append((start - last_s1) / sr)
                last_s1 = start

    avg_durations = {state_names[k]: np.mean(v) for k, v in durations.items() if v}
    std_durations = {state_names[k]: np.std(v) for k, v in durations.items() if v}
    avg_cycle_len = np.mean(cycle_lengths) if cycle_lengths else None
    std_cycle_len = np.std(cycle_lengths) if cycle_lengths else None
    heart_rate = 60.0 / avg_cycle_len if avg_cycle_len else None

    print("\nAnalyseergebnisse:")
    print(f"Herzfrequenz: {heart_rate:.1f} bpm")
    print(f"⌀ Herzzykluslänge: {avg_cycle_len:.3f} s")
    print(f"Zyklusvariabilität: Std = {std_cycle_len:.3f} s | CV = {std_cycle_len / avg_cycle_len:.2%}")

    for name in state_names.values():
        avg = avg_durations.get(name, None)
        std = std_durations.get(name, None)
        if avg is not None:
            print(f"⌀ {name}-Dauer: {avg:.3f} s (± {std:.3f})")


import os
import numpy as np

import os
import numpy as np

import os
import numpy as np


def check_tsv_folder(tsv_folder):
    all_files = [f for f in os.listdir(tsv_folder) if f.endswith(".tsv")]
    valid, invalid = 0, 0

    for f in all_files:
        path = os.path.join(tsv_folder, f)
        try:
            arr = np.genfromtxt(path, delimiter="\t")

            # Bei leerer Datei → ndim == 1 und shape == (0,)
            if arr.ndim == 2 and arr.shape[1] == 3:
                valid += 1
            else:
                print(f"❌ Fehlerhafte TSV: {f} (Shape: {arr.shape})")
                invalid += 1
        except Exception as e:
            print(f"❌ Fehler beim Parsen: {f} → {e}")
            invalid += 1

    print(f"\n📊 Statistik:\n🔢 Gesamt: {len(all_files)}\n✅ Gültig: {valid}\n❌ Defekt: {invalid}")


import os
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
from springer_segmentation.utils import create_segmentation_array


if __name__ == "__main__":
    random.seed(0)
    train_dir = "train_data"
    #train(train_dir, heartrates_from_tsv=False)
    #check_tsv_folder(train_dir)
    # ❌ removed: fehlerhafte 50782_MV_1.tsv mit (Shape: (3,)) --> removed all of 50782
    result = calculate_segmentation('/Users/maximilian/Library/Mobile Documents/com~apple~CloudDocs/7. Semester/Studienarbeit/Online Datasets/Physionet 2016 Challenge/normal/a0007.wav', 4000, 9)
    plot_segmentation_result(result)
    analyze_segmentation(result)
