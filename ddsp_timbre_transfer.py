# DISCLAIMER: Code adapted from the DDSP timbre transfer Colab notebook demo.

#

# Copyright 2020 The DDSP Authors.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import os
import pickle
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

import crepe
import ddsp
import ddsp.training
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf
from pydub import AudioSegment
from scipy.io import wavfile

# Meowify's audio pipeline uses 16 kHz mono audio.

DEFAULT_SAMPLE_RATE = 16000

def detect_notes(loudness_db, f0_confidence, threshold=1.0):
"""
Server-safe replacement for the DDSP Colab detect_notes helper.

```
Detects sections where the input contains an active note based on
loudness and pitch confidence.
"""
loudness_db = np.asarray(loudness_db)
f0_confidence = np.asarray(f0_confidence)

if loudness_db.ndim > 1:
    loudness = np.squeeze(loudness_db)
else:
    loudness = loudness_db

if f0_confidence.ndim > 1:
    confidence = np.squeeze(f0_confidence)
else:
    confidence = f0_confidence

length = min(len(loudness), len(confidence))

loudness = loudness[:length]
confidence = confidence[:length]

# DDSP's note detector uses a loudness threshold and pitch confidence.
# This keeps the same basic behaviour without importing google.colab.
loudness_threshold = np.max(loudness) - 40.0

mask_on = (
    (loudness > loudness_threshold)
    & (confidence > 0.5)
)

# Smooth tiny gaps/noise in the detection.
if len(mask_on) > 2:
    mask_on = np.asarray(mask_on, dtype=bool)

note_on_value = np.clip(
    (loudness - loudness_threshold) / 40.0,
    0.0,
    1.0,
)

return mask_on, note_on_value
```

def fit_quantile_transform(
loudness_db,
mask_on,
inv_quantile=None,
):
"""
Server-safe replacement for the DDSP Colab quantile transform helper.

```
Normalises active-note loudness using the dataset statistics.
"""
loudness_db = np.asarray(loudness_db, dtype=np.float32)
mask_on = np.asarray(mask_on, dtype=bool)

if loudness_db.ndim == 1:
    loudness_db = loudness_db[:, np.newaxis]

if mask_on.ndim > 1:
    mask_on = np.squeeze(mask_on)

mask_on = mask_on[:loudness_db.shape[0]]

if inv_quantile is None:
    return None, loudness_db.copy()

try:
    inv_quantile = np.asarray(inv_quantile)

    # Dataset statistics from DDSP are normally represented as a
    # quantile transform. Reconstruct it with numpy interpolation.
    if inv_quantile.ndim >= 2 and inv_quantile.shape[0] >= 2:
        source = inv_quantile[:, 0]
        target = inv_quantile[:, 1]

        result = loudness_db.copy()

        active = mask_on.astype(bool)

        for channel in range(result.shape[1]):
            result[active, channel] = np.interp(
                result[active, channel],
                source,
                target,
            )

        return None, result

except Exception:
    pass

# Safe fallback if the bundled statistics have an unexpected format.
return None, loudness_db.copy()
```

def get_tuning_factor(f0_midi, f0_confidence, mask_on):
"""
Estimate the tuning offset in semitones.
"""
f0_midi = np.asarray(f0_midi)
f0_confidence = np.asarray(f0_confidence)
mask_on = np.asarray(mask_on, dtype=bool)

```
if f0_midi.ndim > 1:
    f0_midi = np.squeeze(f0_midi)

if f0_confidence.ndim > 1:
    f0_confidence = np.squeeze(f0_confidence)

valid = (
    mask_on
    & np.isfinite(f0_midi)
    & np.isfinite(f0_confidence)
    & (f0_confidence > 0.5)
    & (f0_midi > 0)
)

if not np.any(valid):
    return 0.0

midi = f0_midi[valid]

# Find distance from nearest semitone.
cents = midi - np.round(midi)
return float(np.median(cents))
```

def auto_tune(f0_midi, tuning_factor, mask_on, amount=1.0):
"""
Apply pitch quantisation while preserving the requested amount.
"""
f0_midi = np.asarray(f0_midi, dtype=np.float32).copy()
mask_on = np.asarray(mask_on, dtype=bool)

```
valid = mask_on & np.isfinite(f0_midi) & (f0_midi > 0)

if not np.any(valid):
    return f0_midi

tuned = np.round(f0_midi[valid] - tuning_factor) + tuning_factor

f0_midi[valid] = (
    f0_midi[valid] * (1.0 - amount)
    + tuned * amount
)

return f0_midi
```

def shift_ld(audio_features, ld_shift=0.0):
"""Shift loudness in dB."""
audio_features["loudness_db"] = (
audio_features["loudness_db"] + ld_shift
)
return audio_features

def shift_f0(audio_features, pitch_shift=0.0):
"""Shift pitch by octaves."""
audio_features["f0_hz"] *= 2.0 ** pitch_shift

```
audio_features["f0_hz"] = np.clip(
    audio_features["f0_hz"],
    0.0,
    librosa.midi_to_hz(110.0),
)

return audio_features
```

def audio_file_to_np(
audio_file,
sample_rate=DEFAULT_SAMPLE_RATE,
normalize_db=0.1,
):
"""
Load an audio file as float32 mono audio at the requested sample rate.
"""
if not os.path.exists(audio_file):
raise FileNotFoundError(
"Audio file does not exist: {}".format(audio_file)
)

```
audio = AudioSegment.from_file(audio_file)

# Remove DC offset if supported by the installed pydub version.
try:
    audio = audio.remove_dc_offset()
except Exception:
    pass

if normalize_db is not None:
    try:
        audio = audio.normalize(headroom=normalize_db)
    except Exception:
        pass

# Save to a temporary WAV and load through librosa.
with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav_file:
    temp_wav_file.close()

    audio.export(temp_wav_file.name, format="wav")

    audio_np, _ = librosa.load(
        temp_wav_file.name,
        sr=sample_rate,
        mono=True,
    )

return audio_np.astype(np.float32)
```

def transfer(
audio,
model_dir,
sample_rate=DEFAULT_SAMPLE_RATE,
):
"""
Run DDSP timbre transfer using the bundled Catophone model.
"""
if not os.path.isdir(model_dir):
raise FileNotFoundError(
"Model directory does not exist: {}".format(model_dir)
)

```
audio = np.asarray(audio, dtype=np.float32)

if audio.ndim != 1:
    audio = np.squeeze(audio)

if audio.size == 0:
    raise ValueError("Input audio is empty.")

audio = audio[np.newaxis, :]

# Reset CREPE state before extracting DDSP features.
ddsp.spectral_ops.reset_crepe()

print("Computing DDSP audio features...")

audio_features = ddsp.training.metrics.compute_audio_features(audio)

audio_features["loudness_db"] = (
    audio_features["loudness_db"].astype(np.float32)
)

gin_file = os.path.join(
    model_dir,
    "operative_config-0.gin",
)

dataset_stats_file = os.path.join(
    model_dir,
    "dataset_statistics.pkl",
)

if not os.path.isfile(gin_file):
    raise FileNotFoundError(
        "Missing DDSP gin configuration: {}".format(gin_file)
    )

# Load dataset statistics.
dataset_stats = None

try:
    if tf.io.gfile.exists(dataset_stats_file):
        with tf.io.gfile.GFile(
            dataset_stats_file,
            "rb",
        ) as f:
            dataset_stats = pickle.load(f)
except Exception as err:
    print(
        "Warning: loading dataset statistics failed: {}".format(err)
    )

# Parse the model's gin configuration.
with gin.unlock_config():
    gin.parse_config_file(
        gin_file,
        skip_unknown=True,
    )

# Locate a checkpoint robustly.
checkpoint_files = [
    f
    for f in tf.io.gfile.listdir(model_dir)
    if f.startswith("ckpt-") and f.endswith(".index")
]

if not checkpoint_files:
    raise FileNotFoundError(
        "No DDSP checkpoint found in {}".format(model_dir)
    )

checkpoint_files.sort()

ckpt_name = os.path.splitext(
    checkpoint_files[-1]
)[0]

ckpt = os.path.join(
    model_dir,
    ckpt_name,
)

print("Using DDSP checkpoint: {}".format(ckpt))

# Get model timing parameters.
time_steps_train = int(
    gin.query_parameter(
        "DefaultPreprocessor.time_steps"
    )
)

n_samples_train = int(
    gin.query_parameter(
        "Additive.n_samples"
    )
)

hop_size = int(
    n_samples_train / time_steps_train
)

if hop_size <= 0:
    raise ValueError(
        "Invalid DDSP hop size: {}".format(hop_size)
    )

time_steps = max(
    1,
    int(audio.shape[1] / hop_size),
)

n_samples = time_steps * hop_size

# Override model dimensions to match the supplied audio.
gin_params = [
    "Additive.n_samples = {}".format(n_samples),
    "FilteredNoise.n_samples = {}".format(n_samples),
    "DefaultPreprocessor.time_steps = {}".format(time_steps),
    "oscillator_bank.use_angular_cumsum = True",
]

with gin.unlock_config():
    gin.parse_config(gin_params)

# Trim feature vectors to matching dimensions.
for key in (
    "f0_hz",
    "f0_confidence",
    "loudness_db",
):
    if key in audio_features:
        audio_features[key] = (
            audio_features[key][:time_steps]
        )

audio_features["audio"] = (
    audio_features["audio"][:, :n_samples]
)

# Create and restore the Catophone model.
print("Loading Catophone model...")

model = ddsp.training.models.Autoencoder()

model.restore(ckpt)

# Build the model before inference.
start_time = time.time()

_ = model(
    audio_features,
    training=False,
)

print(
    "DDSP model initialised in {:.2f}s".format(
        time.time() - start_time
    )
)

# Meowify settings from the original demo.
threshold = 1
adjust = True
quiet = 20
autotune = 0
pitch_shift = -1
loudness_shift = 3

audio_features_mod = {
    key: value.copy()
    for key, value in audio_features.items()
}

if adjust and dataset_stats is not None:
    try:
        mask_on, note_on_value = detect_notes(
            audio_features["loudness_db"],
            audio_features["f0_confidence"],
            threshold,
        )

        if np.any(mask_on):
            # Shift the pitch register towards the Catophone dataset.
            target_mean_pitch = dataset_stats["mean_pitch"]

            pitch = ddsp.core.hz_to_midi(
                audio_features["f0_hz"]
            )

            mean_pitch = np.mean(
                pitch[mask_on]
            )

            p_diff = (
                target_mean_pitch - mean_pitch
            )

            p_diff_octave = p_diff / 12.0

            round_fn = (
                np.floor
                if p_diff_octave > 1.5
                else np.ceil
            )

            p_diff_octave = round_fn(
                p_diff_octave
            )

            audio_features_mod = shift_f0(
                audio_features_mod,
                p_diff_octave,
            )

            # Match the Catophone loudness distribution.
            _, loudness_norm = fit_quantile_transform(
                audio_features["loudness_db"],
                mask_on,
                inv_quantile=dataset_stats.get(
                    "quantile_transform"
                ),
            )

            mask_off = np.logical_not(mask_on)

            if loudness_norm.ndim == 1:
                loudness_norm = loudness_norm[:, np.newaxis]

            loudness_norm[mask_off] -= (
                quiet
                * (
                    1.0
                    - note_on_value[
                        mask_off
                    ][:, np.newaxis]
                )
            )

            loudness_norm = np.reshape(
                loudness_norm,
                audio_features[
                    "loudness_db"
                ].shape,
            )

            audio_features_mod[
                "loudness_db"
            ] = loudness_norm

            # Optional autotuning.
            if autotune:
                f0_midi = np.asarray(
                    ddsp.core.hz_to_midi(
                        audio_features_mod["f0_hz"]
                    )
                )

                tuning_factor = get_tuning_factor(
                    f0_midi,
                    audio_features_mod[
                        "f0_confidence"
                    ],
                    mask_on,
                )

                f0_midi_at = auto_tune(
                    f0_midi,
                    tuning_factor,
                    mask_on,
                    amount=autotune,
                )

                audio_features_mod[
                    "f0_hz"
                ] = ddsp.core.midi_to_hz(
                    f0_midi_at
                )

        else:
            print(
                "No notes detected; skipping automatic adjustment."
            )

    except Exception as err:
        # Do not crash the entire web worker just because the optional
        # automatic adjustment cannot be performed.
        print(
            "Warning: automatic adjustment failed: {}".format(
                err
            )
        )

else:
    print(
        "Dataset statistics unavailable; "
        "skipping automatic adjustment."
    )

# Manual shifts used by the original Meowify demo.
audio_features_mod = shift_ld(
    audio_features_mod,
    loudness_shift,
)

audio_features_mod = shift_f0(
    audio_features_mod,
    pitch_shift,
)

print("Generating cat audio...")

return model(
    audio_features_mod,
    training=False,
)
```

def write_to_file(
audio_file,
model_dir,
output,
sample_rate=DEFAULT_SAMPLE_RATE,
):
"""
Convert an input vocal track into Catophone audio and write it as WAV.
"""
if not audio_file:
raise ValueError("No input audio file supplied.")

```
if not model_dir:
    raise ValueError("No Catophone model directory supplied.")

if not output:
    raise ValueError("No output path supplied.")

os.makedirs(
    os.path.dirname(output) or ".",
    exist_ok=True,
)

print(
    "Meowifying {} -> {}".format(
        audio_file,
        output,
    )
)

audio_float = audio_file_to_np(
    audio_file,
    sample_rate=sample_rate,
)

cat_audio_float = transfer(
    audio_float,
    model_dir,
    sample_rate=sample_rate,
)

# DDSP may return [batch, samples].
cat_audio_float = np.asarray(
    cat_audio_float
)

if cat_audio_float.ndim == 2:
    cat_audio_float = cat_audio_float[0]

cat_audio_float = np.nan_to_num(
    cat_audio_float,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# Prevent clipping when converting float audio to int16.
cat_audio_float = np.clip(
    cat_audio_float,
    -1.0,
    1.0,
)

normalizer = float(
    np.iinfo(np.int16).max
)

cat_audio_int = np.asarray(
    cat_audio_float * normalizer,
    dtype=np.int16,
)

wavfile.write(
    output,
    sample_rate,
    cat_audio_int,
)

print(
    "Cat audio written successfully: {}".format(
        output
    )
)
