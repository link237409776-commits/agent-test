from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from .agent import DocumentQaAgent
from .llm_config import load_llm_config, save_llm_settings
from flask import Response
import json

UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.secret_key = os.environ.get("DOCQA_WEB_SECRET", "dev-secret")

# global agent (initialized at startup if configured)
AGENT: DocumentQaAgent | None = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    config = load_llm_config()
    loaded_count = len(AGENT.pdf_paths) if AGENT is not None else 0
    return render_template("index.html", config=config, loaded_count=loaded_count)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    pdf_path = request.form.get("pdf_path", "").strip()

    # handle file upload
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)
        pdf_path = str(save_path)

    # If no pdf_path provided, fall back to preloaded AGENT if available
    global AGENT
    if not pdf_path and AGENT is None:
        flash("请提供 PDF 文件路径或上传 PDF 文件，或启动时自动加载的文档未找到。")
        return redirect(url_for("index"))

    if not question:
        flash("请填写问题。")
        return redirect(url_for("index"))

    # handle optional settings save
    base_url = request.form.get("base_url")
    model = request.form.get("model")
    api_key = request.form.get("api_key")
    temperature = request.form.get("temperature")
    timeout = request.form.get("timeout")
    system_prompt = request.form.get("system_prompt")
    user_prompt_template = request.form.get("user_prompt_template")

    if any(v is not None and v != "" for v in [base_url, model, api_key, temperature, timeout, system_prompt, user_prompt_template]):
        try:
            save_llm_settings({
                "base_url": base_url or "",
                "model": model or "",
                "api_key": api_key or "",
                "temperature": float(temperature) if temperature else 0.1,
                "timeout": float(timeout) if timeout else 60,
                "system_prompt": system_prompt or None,
                "user_prompt_template": user_prompt_template or None,
            })
            flash("LLM 配置已保存到 docqa_agent/llm_settings.json")
        except Exception as exc:
            flash(f"保存配置失败: {exc}")

    # instantiate agent and ask (reuse AGENT if available and no pdf_path provided)
    try:
        if not pdf_path and AGENT is not None:
            agent = AGENT
        else:
            agent = DocumentQaAgent(pdf_path)
            agent.build()
        answer = agent.ask(question)
    except Exception as exc:
        flash(f"处理失败: {exc}")
        return redirect(url_for("index"))

    return render_template("result.html", question=question, answer=answer)


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    question = request.form.get("question", "").strip()
    pdf_path = request.form.get("pdf_path", "").strip()

    # handle file upload
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)
        pdf_path = str(save_path)

    # If no pdf_path provided, allow using the preloaded AGENT
    global AGENT
    if not pdf_path and AGENT is None:
        return Response("data: ERROR\n\ndata: 请提供 PDF 文件路径或上传 PDF 文件。\n\n", mimetype="text/event-stream")

    if not question:
        return Response("data: ERROR\n\ndata: 请填写问题。\n\n", mimetype="text/event-stream")

    # handle optional settings save
    base_url = request.form.get("base_url")
    model = request.form.get("model")
    api_key = request.form.get("api_key")
    temperature = request.form.get("temperature")
    timeout = request.form.get("timeout")
    system_prompt = request.form.get("system_prompt")
    user_prompt_template = request.form.get("user_prompt_template")

    if any(v is not None and v != "" for v in [base_url, model, api_key, temperature, timeout, system_prompt, user_prompt_template]):
        try:
            save_llm_settings({
                "base_url": base_url or "",
                "model": model or "",
                "api_key": api_key or "",
                "temperature": float(temperature) if temperature else 0.1,
                "timeout": float(timeout) if timeout else 60,
                "system_prompt": system_prompt or None,
                "user_prompt_template": user_prompt_template or None,
            })
            # do not flash in stream
        except Exception:
            pass

    def generator():
        yield "data: PROGRESS: Retrieving evidence...\n\n"
        try:
            if not pdf_path and AGENT is not None:
                agent = AGENT
            else:
                agent = DocumentQaAgent(pdf_path)
                agent.build()
            evidences = agent.retrieve(question)
        except Exception as exc:
            yield f"data: ERROR: Failed to build agent or retrieve: {str(exc)}\n\n"
            yield "data: DONE\n\n"
            return

        # do not stream evidences; only stream the LLM-generated answer in chunks
        yield "data: PROGRESS: Calling LLM...\n\n"
        try:
            answer = agent.ask(question)
        except Exception as exc:
            yield f"data: ERROR: LLM call failed: {str(exc)}\n\n"
            yield "data: DONE\n\n"
            return

        text = getattr(answer, "answer", "") or ""

        # chunk the answer to simulate/stream incremental output
        def chunks(s: str, size: int = 300):
            for i in range(0, len(s), size):
                yield s[i : i + size]

        first = True
        for piece in chunks(text, size=300):
            # send each piece as an ANSWER chunk
            payload = {"answer": piece, "first": first}
            yield f"data: ANSWER: {json.dumps(payload, ensure_ascii=False)}\n\n"
            first = False

        yield "data: DONE\n\n"

    return Response(generator(), mimetype="text/event-stream")


def run(host: str = "0.0.0.0", port: int = 5000, debug: bool = True) -> None:
    import webbrowser, threading

    def _open_browser():
        webbrowser.open(f"http://{host}:{port}")

    # initialize agent at startup if a `doc` directory exists (or use DOCQA_DEFAULT_DOC)
    default_doc = os.environ.get("DOCQA_DEFAULT_DOC", "doc")
    try:
        if Path(default_doc).exists():
            try:
                agent = DocumentQaAgent(default_doc)
                agent.build()
                global AGENT
                AGENT = agent
                app.logger.info(f"Auto-loaded {len(agent.pdf_paths)} PDF(s) from {default_doc}")
            except Exception as exc_init:
                app.logger.warning(f"Failed to auto-load documents from {default_doc}: {exc_init}")
    except Exception:
        pass

    threading.Timer(1.0, _open_browser).start()
    app.run(host=host, port=port, debug=debug)


@app.route("/status", methods=["GET"])
def status():
    global AGENT
    if AGENT is None:
        return {"loaded": False, "loaded_count": 0}
    return {"loaded": True, "loaded_count": len(AGENT.pdf_paths), "paths": [str(p) for p in AGENT.pdf_paths]}


if __name__ == "__main__":
    run(host=os.environ.get("DOCQA_WEB_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5000)))
