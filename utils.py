import os
import shutil
import subprocess


SEPARATED_FOLDER = "separated"
OUTPUT_FOLDER = "outputs"


def run_command(command, name="command"):
    print("========================================", flush=True)
    print("RUNNING:", name, flush=True)
    print("========================================", flush=True)
    print(" ".join(command), flush=True)

    # Force CPU execution.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    # Keep TensorFlow from creating excessive CPU threads.
    env["OMP_NUM_THREADS"] = "2"
    env["TF_NUM_INTRAOP_THREADS"] = "2"
    env["TF_NUM_INTEROP_THREADS"] = "2"

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    except Exception as e:
        print("SUBPROCESS COULD NOT START", flush=True)
        print(type(e).__name__, flush=True)
        print(repr(e), flush=True)
        raise

    print("========================================", flush=True)
    print(name, "FINISHED", flush=True)
    print("Exit code:", result.returncode, flush=True)
    print("========================================", flush=True)

    if result.stdout:
        print(result.stdout, flush=True)
    else:
        print("(No output from process)", flush=True)

    # Positive exit code = normal program failure.
    if result.returncode > 0:
        raise RuntimeError(
            f"{name} failed with exit code "
            f"{result.returncode}.\n\n"
            f"{result.stdout}"
        )

    # Negative exit code = process was killed by a signal.
    if result.returncode < 0:
        signal_number = abs(result.returncode)

        raise RuntimeError(
            f"{name} was terminated by signal "
            f"{signal_number}.\n\n"
            f"{result.stdout}"
        )

    return result


def convert_to_wav(input_file, output_file):
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            output_file,
        ],
        name="FFmpeg conversion",
    )


def separate_audio(input_file, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    run_command(
        [
            "spleeter",
            "separate",
            "-p",
            "spleeter:2stems",
            "-i",
            input_file,
            "-o",
            output_directory,
        ],
        name="Spleeter",
    )


def mix_audio(meows, accompaniment, output):
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            meows,
            "-i",
            accompaniment,
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0",
            "-ar",
            "44100",
            "-ac",
            "2",
            output,
        ],
        name="Final audio mix",
    )


def cleanup_job(job_id, input_file=None):
    print("Cleaning up temporary files...", flush=True)

    paths = [
        os.path.join(
            SEPARATED_FOLDER,
            job_id + ".wav"
        ),

        os.path.join(
            SEPARATED_FOLDER,
            job_id
        ),
    ]

    if input_file:
        paths.append(input_file)

    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(
                    path,
                    ignore_errors=True
                )
                print(
                    "Removed directory:",
                    path,
                    flush=True
                )

            elif os.path.exists(path):
                os.remove(path)
                print(
                    "Removed file:",
                    path,
                    flush=True
                )

        except Exception as e:
            print(
                "Cleanup failed for",
                path,
                repr(e),
                flush=True
            )


def process_audio(input_file, job_id):
    os.makedirs(
        SEPARATED_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    working_file = os.path.join(
        SEPARATED_FOLDER,
        job_id + ".wav"
    )

    separation_directory = os.path.join(
        SEPARATED_FOLDER,
        job_id
    )

    try:
        # ========================================
        # STEP 1: CONVERT
        # ========================================

        print("========================================", flush=True)
        print("STEP 1: CONVERTING AUDIO", flush=True)
        print("========================================", flush=True)

        convert_to_wav(
            input_file,
            working_file
        )

        if not os.path.exists(working_file):
            raise FileNotFoundError(
                "FFmpeg did not create the WAV file."
            )

        print(
            "WAV created:",
            working_file,
            flush=True
        )

        # ========================================
        # STEP 2: SPLEETER
        # ========================================

        print("========================================", flush=True)
        print("STEP 2: RUNNING SPLEETER", flush=True)
        print("========================================", flush=True)

        separate_audio(
            working_file,
            separation_directory
        )

        # Spleeter creates:
        #
        # separation_directory/
        #     job_id/
        #         vocals.wav
        #         accompaniment.wav

        base_name = os.path.splitext(
            os.path.basename(working_file)
        )[0]

        song_directory = os.path.join(
            separation_directory,
            base_name
        )

        vocals = os.path.join(
            song_directory,
            "vocals.wav"
        )

        accompaniment = os.path.join(
            song_directory,
            "accompaniment.wav"
        )

        print(
            "Looking for vocals:",
            vocals,
            flush=True
        )

        print(
            "Looking for accompaniment:",
            accompaniment,
            flush=True
        )

        if not os.path.exists(vocals):
            raise FileNotFoundError(
                "Spleeter finished but did not produce "
                "vocals.wav"
            )

        if not os.path.exists(accompaniment):
            raise FileNotFoundError(
                "Spleeter finished but did not produce "
                "accompaniment.wav"
            )

        print(
            "Spleeter separation successful!",
            flush=True
        )

        # ========================================
        # STEP 3: CATOPHONE
        # ========================================

        print("========================================", flush=True)
        print("STEP 3: RUNNING CATOPHONE / DDSP", flush=True)
        print("========================================", flush=True)

        meows = os.path.join(
            OUTPUT_FOLDER,
            job_id + "_meows.wav"
        )

        # IMPORTANT:
        #
        # DDSP is imported HERE rather than when
        # Gunicorn starts.
        #
        # This means Spleeter doesn't have to share
        # memory with the DDSP/TensorFlow models.

        from ddsp_timbre_transfer import write_to_file

        print(
            "Starting Catophone...",
            flush=True
        )

        write_to_file(
            vocals,
            meows
        )

        if not os.path.exists(meows):
            raise FileNotFoundError(
                "Catophone did not produce an output file."
            )

        print(
            "Catophone finished:",
            meows,
            flush=True
        )

        # ========================================
        # STEP 4: MIX
        # ========================================

        print("========================================", flush=True)
        print("STEP 4: MIXING AUDIO", flush=True)
        print("========================================", flush=True)

        output = os.path.join(
            OUTPUT_FOLDER,
            job_id + "_meowified.wav"
        )

        mix_audio(
            meows,
            accompaniment,
            output
        )

        if not os.path.exists(output):
            raise FileNotFoundError(
                "Final output was not created."
            )

        print("========================================", flush=True)
        print("MEOWIFY COMPLETE", flush=True)
        print("========================================", flush=True)
        print("Final output:", output, flush=True)

        return output

    finally:
        # Don't delete the final output.
        #
        # Everything used to create it can go.
        cleanup_job(
            job_id,
            input_file
        )
