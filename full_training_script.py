import sys
import random
import time

from duration_distributions import DataDistribution
from segmentation_model import SegmentationModel
from utils import get_wavs_and_tsvs, get_heart_rate_from_tsv, create_segmentation_array, create_train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import librosa

def train(data_dir, heartrates_from_tsv=False):
    # Get training recordings and segmentations
    train_recordings, train_segmentations, \
        test_recordings, test_segmentations = create_train_test_split(directory=data_dir,
                                                                      frac_train = 0.74,
                                                                      max_train_size=750,
                                                                      max_test_size=100)

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
    model.save("saved_model.pkl")
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

        accuracies[idx] = (seg == annotation).mean()
        weights[idx] = seg.shape[0]
        idx += 1
    print(f"average accuracy: {accuracies.mean()}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights)}")

def predict_wav_clip(model, wav_path, target_sr=4000, clip_duration_sec=6):
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    signal, sr = librosa.load(wav_path, sr=target_sr)
    clip_len = target_sr * clip_duration_sec
    signal = signal[:clip_len]
    if len(signal) < clip_len:
        signal = np.pad(signal, (0, clip_len - len(signal)))

    prediction = model.predict(signal)

    # Upsample prediction to match signal length
    from utils import upsample_states
    upsampled = upsample_states(prediction, old_fs=50, new_fs=target_sr, new_length=len(signal))

    # Plot
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, clip_duration_sec, len(signal))
    plt.plot(time, signal, label="Waveform", alpha=0.7)
    plt.plot(time, upsampled / 4 * np.max(signal), label="Segmentation (states)", linewidth=2)
    plt.title("Waveform with Predicted Segmentation Overlay")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / State")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return prediction


if __name__ == "__main__":
    random.seed(0)
    train_dir = "train_data"
    #train_dir = sys.argv[1]
    #test_dir = sys.argv[2]
    # train(train_dir, heartrates_from_tsv=False)
    # Load pretrained model instead of retraining
    model = SegmentationModel.load("saved_model.pkl")
    #predict_wav_clip(model, "test_data/max_erb_point_5.wav")
    predict_wav_clip(model, "test_data/robert_erb_point.wav")
