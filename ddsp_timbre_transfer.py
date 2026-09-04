import os
import pickle
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

from pydub import AudioSegment
from scipy.io import wavfile


DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE


def get_tuning_factor(f0_midi, f0_confidence, mask_on):
    """Get an offset to the most consistent chromatic intervals."""

    tuning_factors = np.linspace(-0.5, 0.5, 101)

    midi_diffs = (
        f0_midi[mask_on][:, np.newaxis]
        - tuning_factors[np.newaxis, :]
    ) % 1.0

    midi_diffs[midi_diffs > 0.5] -= 1.0

    weights = f0_confidence[mask_on][:, np.newaxis]

    cost_diffs = np.abs(midi_diffs)
    cost_diffs = np.mean(
        weights * cost_diffs,
        axis=0,
    )

    f0_at = (
        f0_midi[mask_on][:, np.newaxis]
        - midi_diffs
    )

    f0_at_diffs = np.diff(
        f0_at,
        axis=0,
    )

    deltas = (
        f0_at_diffs != 0.0
    ).astype(float)

    cost_deltas = np.mean(
        weights[:-1] * deltas,
        axis=0,
    )

    def normalize(value):
        std = np.std(value)

        if std == 0:
            return value - np.mean(value)

        return (
            value - np.mean(value)
        ) / std

    cost = (
        normalize(cost_deltas)
        + normalize(cost_diffs)
    )

    return tuning_factors[np.argmin(cost)]


def auto_tune(
    f0_midi,
    tuning_factor,
    mask_on,
    amount=0.0,
    chromatic=False,
):
    """Reduce variance of f0 from chromatic or scale intervals."""

    if chromatic:
        midi_diff = (
            f0_midi - tuning_factor
        ) % 1.0

        midi_diff[midi_diff > 0.5] -= 1.0

    else:
        major_scale = np.ravel([
            np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i
            for i in range(10)
        ])

        all_scales = np.stack([
            major_scale + i
            for i in range(12)
        ])

        f0_on = f0_midi[mask_on]

        f0_diff_tsn = (
            f0_on[:, np.newaxis, np.newaxis]
            - all_scales[np.newaxis, :, :]
        )

        f0_diff_ts = np.min(
            np.abs(f0_diff_tsn),
            axis=-1,
        )

        f0_diff_s = np.mean(
            f0_diff_ts,
            axis=0,
        )

        scale_idx = np.argmin(f0_diff_s)

        scale = [
            "C", "Db", "D", "Eb", "E", "F",
            "Gb", "G", "Ab", "A", "Bb", "B", "C"
        ][scale_idx]

        f0_diff_tn = (
            f0_midi[:, np.newaxis]
            - all_scales[scale_idx][np.newaxis, :]
        )

        note_idx = np.argmin(
            np.abs(f0_diff_tn),
            axis=-1,
        )

        midi_diff = np.take_along_axis(
            f0_diff_tn,
            note_idx[:, np.newaxis],
            axis=-1,
        )[:, 0]

        print(
            "Autotuning...\n"
            "Inferred key: {}\n"
            "Tuning offset: {} cents".format(
                scale,
                int(tuning_factor * 100),
            )
        )

    return f0_midi - amount * midi_diff


def detect_notes(
    loudness_db,
    f0_confidence,
    threshold=1.0,
):
    """Detect note-on regions from loudness and pitch confidence."""

    loudness_db = np.asarray(
        loudness_db
    ).reshape(-1)

    f0_confidence = np.asarray(
        f0_confidence
    ).reshape(-1)

    note_on_value = np.maximum(
        0.0,
        loudness_db - threshold,
    )

    confidence_mask = (
        f0_confidence > 0.0
    )

    mask_on = (
        note_on_value > 0.0
    ) & confidence_mask

    return mask_on, note_on_value


def fit_quantile_transform(
    loudness_db,
    mask_on,
    inv_quantile=None,
):
    """Match loudness distribution to dataset statistics."""

    loudness_db = np.asarray(
        loudness_db
    )

    flat = loudness_db.reshape(-1)
    mask = np.asarray(mask_on).reshape(-1)

    if inv_quantile is None:
        return None, loudness_db.copy()

    try:
        source = np.sort(
            flat[mask]
        )

        if len(source) == 0:
            return None, loudness_db.copy()

        target = np.asarray(
            inv_quantile
        )

        if target.ndim == 0:
            return None, loudness_db.copy()

        source_quantiles = np.linspace(
            0.0,
            1.0,
            len(source),
        )

        target_quantiles = np.linspace(
            0.0,
            1.0,
            len(target),
        )

        quantiles = np.interp(
            flat,
            source,
            source_quantiles,
        )

        transformed = np.interp(
            quantiles,
            target_quantiles,
            target,
        )

        result = transformed.reshape(
            loudness_db.shape
        )

        return None, result

    except Exception as exc:
        print(
            "Quantile transform failed: {}".format(
                exc
            )
        )

        return None, loudness_db.copy()


def audio_file_to_np(
    audio_file,
    sample_rate=DEFAULT_SAMPLE_RATE,
    normalize_db=0.1,
):
    audio = AudioSegment.from_file(
        audio_file
    )

    audio = audio.remove_dc_offset()

    if normalize_db is not None:
        audio = audio.normalize(
            headroom=normalize_db
        )

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as temp_wav_file:

        filename = temp_wav_file.name

        audio.export(
            filename,
            format="wav",
        )

        audio_np, _ = librosa.load(
            filename,
            sr=sample_rate,
            mono=True,
        )

    return audio_np.astype(
        np.float32
    )


def shift_ld(
    audio_features,
    ld_shift=0.0,
):
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(
    audio_features,
    pitch_shift=0.0,
):
    audio_features["f0_hz"] *= (
        2.0 ** pitch_shift
    )

    audio_features["f0_hz"] = np.clip(
        audio_features["f0_hz"],
        0.0,
        librosa.midi_to_hz(110.0),
    )

    return audio_features


def transfer(
    audio,
    model_dir,
    sample_rate=DEFAULT_SAMPLE_RATE,
):
    audio = audio[np.newaxis, :]

    ddsp.spectral_ops.reset_crepe()

    audio_features = (
        ddsp.training.metrics.compute_audio_features(
            audio
        )
    )

    audio_features["loudness_db"] = (
        audio_features["loudness_db"]
        .astype(np.float32)
    )

    gin_file = os.path.join(
        model_dir,
        "operative_config-0.gin",
    )

    dataset_stats_file = os.path.join(
        model_dir,
        "dataset_statistics.pkl",
    )

    if not tf.io.gfile.exists(gin_file):
        raise FileNotFoundError(
            "Catophone gin configuration not found: {}".format(
                gin_file
            )
        )

    if not tf.io.gfile.exists(
        dataset_stats_file
    ):
        raise FileNotFoundError(
            "Catophone dataset statistics not found: {}".format(
                dataset_stats_file
            )
        )

    try:
        with tf.io.gfile.GFile(
            dataset_stats_file,
            "rb",
        ) as file_handle:
            dataset_stats = pickle.load(
                file_handle
            )
    except Exception as exc:
        print(
            "Loading dataset statistics failed: {}".format(
                exc
            )
        )
        dataset_stats = None

    with gin.unlock_config():
        gin.parse_config_file(
            gin_file,
            skip_unknown=True,
        )

    ckpt_files = [
        filename
        for filename in tf.io.gfile.listdir(
            model_dir
        )
        if filename.startswith("ckpt-")
        and (
            filename.endswith(".index")
            or ".data-" in filename
        )
    ]

    if not ckpt_files:
        raise FileNotFoundError(
            "No Catophone checkpoint found in {}".format(
                model_dir
            )
        )

    checkpoint_names = sorted([
        filename.split(".")[0]
        for filename in ckpt_files
    ])

    ckpt_name = checkpoint_names[-1]

    ckpt = os.path.join(
        model_dir,
        ckpt_name,
    )

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

    time_steps = int(
        audio.shape[1] / hop_size
    )

    n_samples = (
        time_steps * hop_size
    )

    if time_steps <= 0:
        raise ValueError(
            "Audio is too short for Catophone processing."
        )

    gin_params = [
        "Additive.n_samples = {}".format(
            n_samples
        ),
        "FilteredNoise.n_samples = {}".format(
            n_samples
        ),
        "DefaultPreprocessor.time_steps = {}".format(
            time_steps
        ),
        "oscillator_bank.use_angular_cumsum = True",
    ]

    with gin.unlock_config():
        gin.parse_config(
            gin_params
        )

    for key in [
        "f0_hz",
        "f0_confidence",
        "loudness_db",
    ]:
        audio_features[key] = (
            audio_features[key][:time_steps]
        )

    audio_features["audio"] = (
        audio_features["audio"][:, :n_samples]
    )

    model = (
        ddsp.training.models.Autoencoder()
    )

    model.restore(ckpt)

    start_time = time.time()

    # Build the model.
    _ = model(
        audio_features,
        training=False,
    )

    print(
        "Catophone model loaded in {:.2f}s".format(
            time.time() - start_time
        )
    )

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

    if (
        adjust
        and dataset_stats is not None
        and "mean_pitch" in dataset_stats
        and "quantile_transform" in dataset_stats
    ):
        mask_on, note_on_value = detect_notes(
            audio_features["loudness_db"],
            audio_features["f0_confidence"],
            threshold,
        )

        if np.any(mask_on):
            target_mean_pitch = (
                dataset_stats["mean_pitch"]
            )

            pitch = ddsp.core.hz_to_midi(
                audio_features["f0_hz"]
            )

            mean_pitch = np.mean(
                pitch[mask_on]
            )

            pitch_difference = (
                target_mean_pitch
                - mean_pitch
            )

            pitch_difference_octave = (
                pitch_difference / 12.0
            )

            round_fn = (
                np.floor
                if pitch_difference_octave > 1.5
                else np.ceil
            )

            pitch_difference_octave = (
                round_fn(
                    pitch_difference_octave
                )
            )

            audio_features_mod = shift_f0(
                audio_features_mod,
                pitch_difference_octave,
            )

            _, loudness_norm = (
                fit_quantile_transform(
                    audio_features["loudness_db"],
                    mask_on,
                    inv_quantile=dataset_stats[
                        "quantile_transform"
                    ],
                )
            )

            mask_off = np.logical_not(
                mask_on
            )

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

            if autotune:
                f0_midi = np.array(
                    ddsp.core.hz_to_midi(
                        audio_features_mod[
                            "f0_hz"
                        ]
                    )
                )

                tuning_factor = (
                    get_tuning_factor(
                        f0_midi,
                        audio_features_mod[
                            "f0_confidence"
                        ],
                        mask_on,
                    )
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
                "Skipping auto-adjust: no notes detected."
            )

    else:
        print(
            "Skipping auto-adjust: dataset statistics unavailable."
        )

    audio_features_mod = shift_ld(
        audio_features_mod,
        loudness_shift,
    )

    audio_features_mod = shift_f0(
        audio_features_mod,
        pitch_shift,
    )

    return model(
        audio_features_mod,
        training=False,
    )


def write_to_file(
    audio_file,
    model_dir,
    output,
    sample_rate=DEFAULT_SAMPLE_RATE,
):
    os.makedirs(
        os.path.dirname(
            os.path.abspath(output)
        ),
        exist_ok=True,
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

    if len(cat_audio_float.shape) == 2:
        cat_audio_float = cat_audio_float[0]

    cat_audio_float = np.nan_to_num(
        cat_audio_float,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cat_audio_float = np.clip(
        cat_audio_float,
        -1.0,
        1.0,
    )

    normalizer = float(
        np.iinfo(np.int16).max
    )

    cat_audio_int = np.array(
        cat_audio_float * normalizer,
        dtype=np.int16,
    )

    wavfile.write(
        output,
        sample_rate,
        cat_audio_int,
    )

