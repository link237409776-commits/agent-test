from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.llm_config import load_llm_config, save_llm_settings

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.secret_key = os.environ.get("DOCQA_WEB_SECRET", "dev-secret")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    config = load_llm_config()
    return render_template("index.html", config=config)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    pdf_path = request.form.get("pdf_path", "").strip()

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

    # handle file upload
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)
        pdf_path = str(save_path)

    if not pdf_path:
        flash("请提供 PDF 文件路径或上传 PDF 文件。")
        return redirect(url_for("index"))

    if not question:
        flash("请填写问题。")
        return redirect(url_for("index"))

    # instantiate agent and ask
    try:
        agent = DocumentQaAgent(pdf_path)
        agent.build()
        answer = agent.ask(question)
    except Exception as exc:
        flash(f"处理失败: {exc}")
        return redirect(url_for("index"))

    return render_template("result.html", question=question, answer=answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
