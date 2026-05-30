# 智能文档问答 Agent（已同步当前实现）

本仓库实现了一个以证据优先的 PDF 问答原型：先构建 TF‑IDF 索引检索证据，再在可用时调用大模型进行增强生成；未配置或调用失败时回退到基于证据的摘录答案。

## 快速开始（Web 模式）

1. 安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. 启动 Web 服务（开发模式，默认会尝试自动加载仓库下的 `doc` 目录）：

```powershell
$env:DOCQA_RUN_MODE='web'
python -m docqa_agent
```

可通过环境变量 `DOCQA_DEFAULT_DOC` 指定启动时自动加载的文档目录（默认 `doc`）。

3. 访问页面：

- http://127.0.0.1:5000

页面只展示：模型地址（Base URL）、模型名称、API Key、提问输入和实时输出。其余设置（temperature、timeout、system_prompt 等）以隐藏字段保留，写入 `docqa_agent/llm_settings.json`（如果用户保存）。

4. 状态检查：

```powershell
curl http://127.0.0.1:5000/status
```

该接口返回是否已自动加载索引以及加载的 PDF 路径列表。

## 命令行与交互（保留）

仍然支持 CLI 用法用于离线/批处理：

```bash
python -m docqa_agent.cli path/to/file.pdf -q "你的问题"
```

或者交互式：

```bash
python -m docqa_agent.cli path/to/file.pdf
```

## 配置（LLM）

- 配置文件：`docqa_agent/llm_settings.json`（优先读取）。
- 环境变量回退：`DOCQA_LLM_BASE_URL`、`DOCQA_LLM_MODEL`、`DOCQA_LLM_API_KEY` 等。
- 当未配置 `base_url`/`model` 时，系统会跳过调用 LLM，仅返回基于检索的摘录回答。

## 流式输出行为

- 前端通过 `/ask_stream` 接收服务端事件（SSE 风格）。
- 为减少噪音，服务端仅向前端逐块发送最终答案内容（后端将完整答案按块切分并逐步发送），不再逐条推送证据或自检信息。
- 如果需要真实的 token 级别流式转发，可在支持的模型 API 基础上改造 `generate_with_llm`。

## 存储与上传

- 页面上传的 PDF 将保存到 `uploads/` 目录，后端会在处理请求时使用该文件。
- 启动时自动加载的 `doc` 目录用于构建全局索引（被 `web_app` 的全局 `AGENT` 复用）。

## 测试

```bash
pytest -q
```

## 重要实现位置

- 索引与检索：`docqa_agent/vector_store.py`、`docqa_agent/agent.py`
- 策略与答案构建：`docqa_agent/answerer.py`
- LLM 配置与持久化：`docqa_agent/llm_config.py`（`llm_settings.json`）
- Web 服务与流式接口：`docqa_agent/web_app.py`、`templates/index.html`

## 注意事项与改进建议

- `llm_settings.json` 包含敏感密钥，生产环境请通过秘钥管理或加密存储替代。
- 当前流式实现是在后端将完整答案切分后发送；若需无缝 token 流式显示，需要模型提供流式 API 并在 `generate_with_llm` 中逐 token转发。
- 大依赖（如 `paddlepaddle`、`paddleocr`）体积大，按需安装或在容器/虚拟环境中管理。

如果你希望我把 README 中的某一段调整得更详细或恢复某些示例命令（例如增加 Windows 特定示例），告诉我要修改的部分。
