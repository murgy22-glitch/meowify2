import os
import re
import subprocess

from yt_dlp import YoutubeDL
from spleeter.separator import Separator

from ddsp_timbre_transfer import write_to_file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static")
WORK_DIR = os.path.join(BASE_DIR, "work")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)


def verify(url):
    pattern = re.compile(
        r"^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$",
        re.IGNORECASE,
    )
    return bool(pattern.match(url))


def safe_filename(value):
    value = str(value)

    # Remove characters that are unsafe in Linux filenames.
    value = re.sub(r"[^\w\-. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")

    if not value:
        value = "meowify_audio"

    return value[:120]


def download(session):
    url = session.get("requested_url")

    if not url:
        raise ValueError("No YouTube URL was supplied.")

    output_template = os.path.join(
        WORK_DIR,
        "%(title).120B-%(id)s.%(ext)s"
    )

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 3,
        "fragment_retries": 3,

        # Current yt-dlp YouTube extraction setup.
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "-android_sdkless"]
            }
        },
    }

    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)

    title = info.get("title") or "Meowify"
    video_id = info.get("id")

    if not video_id:
        raise RuntimeError("YouTube did not return a video ID.")

    filename_base = safe_filename(
        "{}-{}".format(title, video_id)
    )

    # Locate the actual downloaded file.
    downloaded = ydl.prepare_filename(info)

    if not os.path.exists(downloaded):
        # yt-dlp may have changed the extension during post-processing.
        candidates = []

        for filename in os.listdir(WORK_DIR):
            if video_id in filename:
                candidates.append(
                    os.path.join(WORK_DIR, filename)
                )

        if not candidates:
            raise FileNotFoundError(
                "yt-dlp reported success but the downloaded file "
                "could not be found."
            )

        downloaded = candidates[0]

    extension = os.path.splitext(downloaded)[1].lstrip(".") or "webm"

    session["title"] = title
    session["ext"] = extension
    session["filename"] = filename_base
    session["source_file"] = downloaded


def run_command(command):
    print("Running:", " ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {}:\n{}".format(
                result.returncode,
                result.stdout,
            )
        )


def split_vocals(session):
    f = session.get("filename")
    source_file = session.get("source_file")

    if not f or not source_file:
        raise RuntimeError("Downloaded audio information is missing.")

    wav_file = os.path.join(WORK_DIR, f + ".wav")

    if not os.path.exists(wav_file):
        run_command([
            "ffmpeg",
            "-y",
            "-i",
            source_file,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_file,
        ])

    split_root = os.path.join(WORK_DIR, "split")

    vocals = os.path.join(
        split_root,
        f,
        "vocals.wav",
    )

    accompaniment = os.path.join(
        split_root,
        f,
        "accompaniment.wav",
    )

    if not os.path.exists(vocals) or not os.path.exists(accompaniment):
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(
            wav_file,
            split_root,
        )

        if not os.path.exists(vocals):
            raise FileNotFoundError(
                "Spleeter did not create vocals.wav."
            )

        if not os.path.exists(accompaniment):
            raise FileNotFoundError(
                "Spleeter did not create accompaniment.wav."
            )

        convert_samplerate(vocals)
        convert_samplerate(accompaniment)

    session["vocals"] = vocals
    session["acc"] = accompaniment


def meowify(session):
    f = session.get("filename")
    vocals = session.get("vocals")

    if not f or not vocals:
        raise RuntimeError("Vocal track is missing.")

    meows = os.path.join(
        WORK_DIR,
        "split",
        f,
        "meows.wav",
    )

    if not os.path.exists(meows):
        write_to_file(
            vocals,
            "catophone",
            meows,
        )

    if not os.path.exists(meows):
        raise FileNotFoundError(
            "Catophone did not create meows.wav."
        )

    session["meows"] = meows


def merge_meows_and_music(session):
    f = session.get("filename")
    meows = session.get("meows")
    accompaniment = session.get("acc")

    if not f or not meows or not accompaniment:
        raise RuntimeError("Audio tracks required for mixing are missing.")

    final = os.path.join(
        OUTPUT_DIR,
        f + "-final.wav",
    )

    if not os.path.exists(final):
        run_command([
            "ffmpeg",
            "-y",
            "-i",
            meows,
            "-i",
            accompaniment,
            "-filter_complex",
            "amix=inputs=2:duration=longest",
            "-ac",
            "2",
            final,
        ])

    if not os.path.exists(final):
        raise FileNotFoundError(
            "FFmpeg did not create the final Meowify file."
        )

    session["final"] = final


def convert_samplerate(file_path, sample_rate="16k"):
    directory, filename = os.path.split(file_path)

    temporary = os.path.join(
        directory,
        "_" + filename,
    )

    run_command([
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        sample_rate,
        temporary,
    ])

    os.replace(temporary, file_path)

