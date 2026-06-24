from flask import Flask, request, render_template, send_from_directory
import os, logging
from video_translator import process_local_video, process_youtube_video

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        lang_codes     = request.form.getlist("languages")
        input_type     = request.form.get("input_type", "youtube")
        voice          = request.form.get("voice", "male")
        download_video = request.form.get("download_video", "1") == "1"

        try:
            if input_type == "youtube":
                youtube_url = request.form.get("youtube_url", "").strip()
                result = process_youtube_video(youtube_url, lang_codes,
                                               voice=voice,
                                               download_video=download_video)
            else:
                uploaded_file = request.files.get("video_file")
                if not uploaded_file or uploaded_file.filename == "":
                    error = "No video file selected. Please choose a file."
                    return render_template("index.html", error=error)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                        uploaded_file.filename)
                uploaded_file.save(filepath)
                log.info("[FORENSIC] Uploaded file saved to: %s", filepath)
                log.info("[FORENSIC] Absolute path: %s", os.path.abspath(filepath))
                log.info("[FORENSIC] File size: %d bytes", os.path.getsize(filepath))
                result = process_local_video(filepath, lang_codes, voice=voice)

            # FORENSIC: Log returned result
            log.info("[FORENSIC] Result received from pipeline")
            if result.get("output_files"):
                for f in result["output_files"]:
                    log.info("[FORENSIC] Output file: %s | Path: %s", 
                             f.get("file_name"), f.get("file_path"))
            log.info("[FORENSIC] Run timestamp: %s", result.get("run_timestamp"))
            
            return render_template("result.html", result=result)

        except RuntimeError as exc:
            error = str(exc)
            log.error("Processing error: %s", error)
        except Exception as exc:
            error = "An unexpected error occurred. Please try again."
            log.exception("Unexpected: %s", exc)

    return render_template("index.html", error=error)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory("final_outputs", filename, as_attachment=True)


@app.route('/stream/<path:filename>')
def stream_file(filename):
    """Serve files inline so HTML5 players can play them without forcing download."""
    return send_from_directory("final_outputs", filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)