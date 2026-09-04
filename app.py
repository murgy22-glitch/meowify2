import os

from flask import Flask, render_template, request, session

from utils import (
    verify,
    download,
    split_vocals,
    meowify,
    merge_meows_and_music,
)


app = Flask(__name__)

# Use Render's SECRET_KEY environment variable.
app.secret_key = os.environ.get("SECRET_KEY", "development-secret-key")


@app.route("/processing", methods=["GET", "POST"])
def processing():
    if request.method == "GET":
        return render_template(
            "waiting.html",
            img="/static/images/waiting.jpg"
        )

    try:
        download(session)
        split_vocals(session)
        meowify(session)
        merge_meows_and_music(session)

        return "done"

    except Exception as exc:
        app.logger.exception("Meowify processing failed")
        return "Processing failed: {}".format(exc), 500


@app.route("/success", methods=["GET"])
def success():
    return render_template(
        "success.html",
        title=session.get("title"),
        vocals=session.get("final"),
        img="/static/images/singing.jpg",
    )


@app.route("/", methods=["GET", "POST"])
def index():
    title = "Meowify"

    if request.method == "POST":
        requested_url = request.form.get("url", "").strip()
        session["requested_url"] = requested_url

        if not requested_url:
            return render_template(
                "index.html",
                title="That's an empty URL, are you kittying me?",
                img="/static/images/angrier_kitty.jpg",
            )

        if verify(requested_url):
            return render_template(
                "waiting.html",
                img="/static/images/waiting.jpg"
            )

        return render_template(
            "index.html",
            title="That's not a youtube URL, are you kittying me?",
            img="/static/images/angry_kitty.jpg",
        )

    return render_template(
        "index.html",
        title=title,
        img="/static/images/starting.png",
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
