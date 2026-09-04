
import os
import subprocess

from ddsp_timbre_transfer import write_to_file


SEPARATED_FOLDER = "separated"
OUTPUT_FOLDER = "outputs"


def run_command(command):

    print("Running:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n\n"
            + result.stdout
        )

    return result


def convert_to_wav(input_file, output_file):

    run_command([
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ar",
        "16000",
        "-ac",
        "2",
        output_file,
    ])


def separate_audio(input_file, output_directory):

    run_command([
        "spleeter",
        "separate",
        "-p",
        "spleeter:2stems",
        "-i",
        input_file,
        "-o",
        output_directory,
    ])


def mix_audio(meows, accompaniment, output):

    run_command([
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
    ])


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

    print("Converting uploaded audio...")

    convert_to_wav(
        input_file,
        working_file
    )

    separation_directory = os.path.join(
        SEPARATED_FOLDER,
        job_id
    )

    print("Running Spleeter...")

    separate_audio(
        working_file,
        separation_directory
    )

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

    if not os.path.exists(vocals):
        raise FileNotFoundError(
            "Spleeter did not produce vocals.wav"
        )

    if not os.path.exists(accompaniment):
        raise FileNotFoundError(
            "Spleeter did not produce accompaniment.wav"
        )

    meows = os.path.join(
        OUTPUT_FOLDER,
        job_id + "_meows.wav"
    )

    print("Running Catophone...")

    write_to_file(
        vocals,
        meows
    )

    if not os.path.exists(meows):
        raise FileNotFoundError(
            "Catophone did not produce an output file"
        )

    output = os.path.join(
        OUTPUT_FOLDER,
        job_id + "_meowified.wav"
    )

    print("Mixing meows with accompaniment...")

    mix_audio(
        meows,
        accompaniment,
        output
    )

    if not os.path.exists(output):
        raise FileNotFoundError(
            "Final output was not created"
        )

    print("========================================")
    print("MEOWIFY COMPLETE")
    print(output)
    print("========================================")

    return output

