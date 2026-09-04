import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
)

from werkzeug.utils import secure_filename

from utils import process_audio


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "meowify-development-secret"
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "mp3",
    "wav",
    "m4a",
    "flac",
    "ogg",
    "aac",
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "audio" not in request.files:
        return "No audio file uploaded.", 400

    file = request.files["audio"]

    if file.filename == "":
        return "No file selected.", 400

    if not allowed_file(file.filename):
        return (
            "Unsupported file type. "
            "Please upload MP3, WAV, M4A, FLAC, OGG or AAC.",
            400,
        )

    job_id = uuid.uuid4().hex

    extension = file.filename.rsplit(".", 1)[1].lower()

    filename = secure_filename(
        f"{job_id}.{extension}"
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    session["input_file"] = filepath
    session["job_id"] = job_id
    session["original_name"] = file.filename

    return redirect(url_for("waiting"))


@app.route("/waiting")
def waiting():
    return render_template(
        "waiting.html",
        filename=session.get("original_name", "audio")
    )


@app.route("/processing", methods=["POST"])
def processing():

    input_file = session.get("input_file")
    job_id = session.get("job_id")

    if not input_file or not os.path.exists(input_file):
        return "Input file no longer exists.", 400

    if not job_id:
        return "Invalid processing session.", 400

    try:

        output_file = process_audio(
            input_file,
            job_id
        )

        session["output_file"] = output_file

        return "done"

    except Exception as e:

        print("========================================")
        print("MEOWIFY PROCESSING ERROR")
        print("========================================")
        print(e)
        print("========================================")

        return "processing_error", 500


@app.route("/success")
def success():

    if "output_file" not in session:
        return redirect(url_for("index"))

    return render_template(
        "success.html",
        filename=session.get(
            "original_name",
            "your song"
        )
    )


@app.route("/download")
def download():

    output_file = session.get("output_file")

    if not output_file or not os.path.exists(output_file):
        return "Output file not found.", 404

    return send_file(
        output_file,
        as_attachment=True,
        download_name="meowified.wav",
        mimetype="audio/wav",
    )


@app.errorhandler(413)
def too_large(error):
    return (
        "That file is too large. Maximum upload size is 100 MB.",
        413,
    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
