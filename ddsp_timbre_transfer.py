import os
import pickle
import numpy as np
import tensorflow as tf
import ddsp
import ddsp.training
import ddsp.losses
import ddsp.core
import ddsp.spectral_ops
import ddsp.processors
import gin

from ddsp.colab import colab_utils as _unused_colab_utils


# ============================================================
# Server-safe configuration
# ============================================================

DEFAULT_SAMPLE_RATE = 16000


# ============================================================
# Utility functions
# ============================================================

def audio_file_to_np(audio_file,
                     sample_rate=DEFAULT_SAMPLE_RATE,
                     mono=True):
    """
    Load an audio file into a NumPy array.

    This replaces the Google Colab audio-loading helpers used
    by the original DDSP notebook.
    """
    import librosa

    audio, _ = librosa.load(
        audio_file,
        sr=sample_rate,
        mono=mono
    )

    audio = np.asarray(audio, dtype=np.float32)

    if np.any(~np.isfinite(audio)):
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    return audio


def detect_notes(audio,
                 frame_rate=250,
                 threshold=1.0):
    """
    Server-safe replacement for the DDSP Colab detect_notes helper.

    Detects sections where the input contains an active note based
    on the loudness of the audio.
    """
    import librosa

    if audio is None or len(audio) == 0:
        return np.array([], dtype=bool)

    audio = np.asarray(audio, dtype=np.float32)

    if np.any(~np.isfinite(audio)):
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    hop_length = max(
        1,
        int(DEFAULT_SAMPLE_RATE / frame_rate)
    )

    rms = librosa.feature.rms(
        y=audio,
        hop_length=hop_length
    )[0]

    if len(rms) == 0:
        return np.array([], dtype=bool)

    rms_db = librosa.amplitude_to_db(
        rms + 1e-7,
        ref=np.max
    )

    notes = rms_db > -threshold

    return notes


def fit_quantile_transform(values):
    """
    Simple server-safe normalization used instead of the
    original Colab-specific helper.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return values

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low = np.percentile(values, 1)
    high = np.percentile(values, 99)

    if high <= low:
        return np.zeros_like(values)

    values = (values - low) / (high - low)

    return np.clip(values, 0.0, 1.0)


def get_tuning_factor(pitch):
    """
    Estimate the tuning correction from pitch values.
    """
    pitch = np.asarray(pitch, dtype=np.float32)

    pitch = pitch[np.isfinite(pitch)]

    if len(pitch) == 0:
        return 0.0

    median_pitch = np.median(pitch)

    if median_pitch <= 0:
        return 0.0

    midi = 69.0 + 12.0 * np.log2(median_pitch / 440.0)

    nearest_note = np.round(midi)

    cents = (midi - nearest_note) * 100.0

    return float(cents)


def auto_tune(f0_hz):
    """
    Apply a simple automatic tuning correction.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32)

    result = f0_hz.copy()

    valid = np.isfinite(result) & (result > 0)

    if not np.any(valid):
        return result

    midi = np.zeros_like(result)

    midi[valid] = (
        69.0
        + 12.0 * np.log2(result[valid] / 440.0)
    )

    rounded = np.round(midi)

    corrected = (
        440.0
        * 2.0 ** ((rounded - 69.0) / 12.0)
    )

    result[valid] = corrected[valid]

    return result


# ============================================================
# Model helpers
# ============================================================

def find_checkpoint(model_dir):
    """
    Find the latest TensorFlow checkpoint in the model directory.
    """
    index_files = []

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            "Model directory does not exist: {}".format(model_dir)
        )

    for filename in os.listdir(model_dir):
        if filename.endswith(".index"):
            index_files.append(
                os.path.join(
                    model_dir,
                    filename[:-6]
                )
            )

    if not index_files:
        raise FileNotFoundError(
            "No TensorFlow checkpoint found in {}".format(model_dir)
        )

    index_files.sort()

    return index_files[-1]


# ============================================================
# DDSP transfer
# ============================================================

def transfer(audio,
             model_dir,
             output_file,
             threshold=1.0,
             adjust=True,
             quiet=20,
             autotune=0,
             pitch_shift=-1,
             loudness_shift=3):
    """
    Run DDSP timbre transfer using the bundled Catophone model.
    """

    import librosa
    import soundfile as sf

    if not os.path.exists(audio):
        raise FileNotFoundError(
            "Input audio does not exist: {}".format(audio)
        )

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            "Model directory does not exist: {}".format(model_dir)
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio_np = audio_file_to_np(
        audio,
        sample_rate=DEFAULT_SAMPLE_RATE,
        mono=True
    )

    if len(audio_np) == 0:
        raise ValueError("Input audio is empty.")

    # --------------------------------------------------------
    # Load model configuration
    # --------------------------------------------------------

    gin_file = os.path.join(
        model_dir,
        "operative_config-0.gin"
    )

    stats_file = os.path.join(
        model_dir,
        "dataset_statistics.pkl"
    )

    if not os.path.exists(gin_file):
        raise FileNotFoundError(
            "Missing DDSP gin configuration: {}".format(gin_file)
        )

    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            "Missing DDSP dataset statistics: {}".format(stats_file)
        )

    checkpoint = find_checkpoint(model_dir)

    # --------------------------------------------------------
    # Configure DDSP
    # --------------------------------------------------------

    gin.clear_config()

    with gin.unlock_config():
        gin.parse_config_file(gin_file)

    # --------------------------------------------------------
    # Load dataset statistics
    # --------------------------------------------------------

    with open(stats_file, "rb") as f:
        dataset_statistics = pickle.load(f)

    # --------------------------------------------------------
    # Prepare audio
    # --------------------------------------------------------

    audio_np = np.nan_to_num(
        audio_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    audio_np = np.clip(
        audio_np,
        -1.0,
        1.0
    )

    # --------------------------------------------------------
    # DDSP spectral processing
    # --------------------------------------------------------

    audio_tensor = tf.convert_to_tensor(
        audio_np,
        dtype=tf.float32
    )

    audio_tensor = audio_tensor[tf.newaxis, :]

    # The original project uses the DDSP notebook pipeline.
    # Build the standard processor objects from the configured gin.
    try:
        from ddsp.training import models

        model = models.Autoencoder()

    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize the DDSP Autoencoder: {}".format(exc)
        )

    # --------------------------------------------------------
    # Restore checkpoint
    # --------------------------------------------------------

    checkpoint_obj = tf.train.Checkpoint(
        model=model
    )

    checkpoint_obj.restore(
        checkpoint
    ).expect_partial()

    # --------------------------------------------------------
    # Extract DDSP features
    # --------------------------------------------------------

    try:
        from ddsp.training import preprocessing

        processor = preprocessing.F0Loudness()

        features = processor(
            audio_tensor,
            training=False
        )

    except Exception:
        # Fall back to a minimal feature extraction path.
        import librosa

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C7"),
            sr=DEFAULT_SAMPLE_RATE
        )

        loudness = librosa.feature.rms(
            y=audio_np
        )[0]

        f0 = np.nan_to_num(
            f0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        loudness = np.nan_to_num(
            loudness,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        features = {
            "f0_hz": tf.convert_to_tensor(
                f0[np.newaxis, :],
                dtype=tf.float32
            ),
            "loudness_db": tf.convert_to_tensor(
                loudness[np.newaxis, :],
                dtype=tf.float32
            )
        }

    # --------------------------------------------------------
    # Pitch shifting
    # --------------------------------------------------------

    if pitch_shift != 0 and "f0_hz" in features:
        f0 = features["f0_hz"]

        multiplier = 2.0 ** (
            float(pitch_shift) / 12.0
        )

        features["f0_hz"] = (
            f0 * multiplier
        )

    # --------------------------------------------------------
    # Loudness adjustment
    # --------------------------------------------------------

    if (
        loudness_shift != 0
        and "loudness_db" in features
    ):
        features["loudness_db"] = (
            features["loudness_db"]
            + float(loudness_shift)
        )

    # --------------------------------------------------------
    # Automatic tuning
    # --------------------------------------------------------

    if autotune and "f0_hz" in features:
        f0 = features["f0_hz"].numpy()

        f0 = auto_tune(f0)

        features["f0_hz"] = tf.convert_to_tensor(
            f0,
            dtype=tf.float32
        )

    # --------------------------------------------------------
    # Quiet-note suppression
    # --------------------------------------------------------

    if quiet and "loudness_db" in features:
        loudness = features["loudness_db"]

        features["loudness_db"] = tf.where(
            loudness < -float(quiet),
            tf.ones_like(loudness) * -120.0,
            loudness
        )

    # --------------------------------------------------------
    # Generate output
    # --------------------------------------------------------

    try:
        audio_gen = model(
            features,
            training=False
        )

        if isinstance(audio_gen, dict):
            if "audio_gen" in audio_gen:
                audio_gen = audio_gen["audio_gen"]
            elif "audio" in audio_gen:
                audio_gen = audio_gen["audio"]

        generated = audio_gen.numpy()

    except Exception as exc:
        raise RuntimeError(
            "DDSP model inference failed: {}".format(exc)
        )

    # --------------------------------------------------------
    # Clean generated audio
    # --------------------------------------------------------

    generated = np.asarray(
        generated,
        dtype=np.float32
    )

    generated = np.squeeze(
        generated
    )

    generated = np.nan_to_num(
        generated,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    generated = np.clip(
        generated,
        -1.0,
        1.0
    )

    # --------------------------------------------------------
    # Make output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        os.path.abspath(output_file)
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write WAV
    # --------------------------------------------------------

    sf.write(
        output_file,
        generated,
        DEFAULT_SAMPLE_RATE,
        subtype="PCM_16"
    )

    return output_file


# ============================================================
# Public API used by utils.py
# ============================================================

def write_to_file(
        input_file,
        model_name,
        output_file,
        threshold=1,
        adjust=True,
        quiet=20,
        autotune=0,
        pitch_shift=-1,
        loudness_shift=3):
    """
    Convert an input vocal recording into Catophone vocals.

    This is the function called by utils.py.
    """

    model_dir = model_name

    if not os.path.isabs(model_dir):
        model_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            model_dir
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_file)
        ),
        exist_ok=True
    )

    return transfer(
        audio=input_file,
        model_dir=model_dir,
        output_file=output_file,
        threshold=threshold,
        adjust=adjust,
        quiet=quiet,
        autotune=autotune,
        pitch_shift=pitch_shift,
        loudness_shift=loudness_shift
    )
