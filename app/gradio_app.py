"""
Gradio 演示界面 — Agent Platform 可视化交互展示

启动方式：
    python app/gradio_app.py
    python app/gradio_app.py --port 7860 --api-url http://localhost:8000 --share
"""

import argparse
import json
import uuid
from pathlib import Path

import gradio as gr
import httpx


def build_demo(api_url: str = "http://localhost:8000"):
    """构建 Gradio Blocks 界面"""

    # ========== 对话 Tab ==========

    async def load_history(sid: str):
        """从后端 Redis 加载历史对话"""
        if not sid:
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{api_url}/history/{sid}")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                items = data.get("history", [])
                return [
                    {"role": item["role"], "content": item["content"]}
                    for item in items
                ]
        except Exception:
            return []

    async def chat_handler(message: str, history: list, mode: str, sid: str):  # pyright: ignore[reportMissingTypeArgument]
        """
        核心回调 — 用户发送消息后调用
        history 格式: [{"role": "user", "content": "..."}, ...]
        """
        payload = {"mode": mode, "query": message, "session_id": sid}

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{api_url}/agent/run?stream=true", json=payload,
            ) as response:
                streaming = ""
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    event = data.get("event", "")
                    content = data.get("content", "")

                    if event == "done":
                        break
                    elif event == "error":
                        streaming += f"\n\n❌ {content}"
                        break
                    elif event in ("thought_delta", "tool_result", "retrieval"):
                        streaming += f"\n\n> {content}"
                    elif event == "text_delta":
                        streaming += content

                    history[-1]["content"] = streaming.strip() or "…"
                    yield history

        history[-1]["content"] = streaming.strip()
        yield history

    # ========== 知识库 Tab ==========

    async def upload_files(files: list, progress=gr.Progress()):  # pyright: ignore[reportMissingTypeArgument]
        """批量上传文档到知识库"""
        if not files:
            return "请选择文件"

        results: list[str] = []
        for file_path in progress.tqdm(files, desc="上传并解析中"):  # pyright: ignore[reportUnknownVariableType]
            file_path = str(file_path)  # 统一转字符串路径
            filename = Path(file_path).name
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    with open(file_path, "rb") as fh:
                        resp = await client.post(
                            f"{api_url}/documents/upload",
                            files={"file": (filename, fh)},
                        )
                    data = resp.json()
                    if data.get("status") == "ok":
                        results.append(f"✅ {filename} — {data['chunks']} 个文本块")
                    else:
                        results.append(f"❌ {filename} — {data.get('message', '未知错误')}")
            except Exception as e:
                results.append(f"❌ {filename} — {e}")

        return "\n".join(results)

    # ========== UI 布局 ==========

    with gr.Blocks(title="Agent Platform") as demo:
        gr.Markdown("# Agent Platform")
        gr.Markdown("基于 LangGraph + Milvus + DeepSeek 的智能 Agent 服务平台")

        # session_id：显示在 Textbox，刷新后可输入旧 ID 恢复对话
        default_sid = uuid.uuid4().hex

        with gr.Tabs():
            # ---- Tab 1: 对话 ----
            with gr.Tab("💬 对话"):
                mode_selector = gr.Radio(
                    choices=[("ReAct 思考", "react"), ("RAG 快速", "rag")],
                    value="react",
                    label="Agent 模式",
                )

                chatbot = gr.Chatbot(label="对话", height=450)

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入你的问题",
                        placeholder="试试：计算 123 * 456",
                        scale=4,
                    )
                    sid_input = gr.Textbox(
                        label="Session ID（刷新后输入旧 ID 可恢复对话）",
                        value=default_sid,
                        scale=3,
                    )

                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    restore_btn = gr.Button("🔄 恢复对话")
                    clear_btn = gr.Button("清空对话")

                submit_btn.click(
                    fn=chat_handler,
                    inputs=[msg_input, chatbot, mode_selector, sid_input],
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[msg_input])

                msg_input.submit(
                    fn=chat_handler,
                    inputs=[msg_input, chatbot, mode_selector, sid_input],
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[msg_input])

                restore_btn.click(
                    fn=load_history,
                    inputs=[sid_input],
                    outputs=[chatbot],
                )

                clear_btn.click(
                    fn=lambda: ([], uuid.uuid4().hex),
                    outputs=[chatbot, sid_input],
                )

            # ---- Tab 2: 知识库 ----
            with gr.Tab("📚 知识库"):
                gr.Markdown("### 上传文档到知识库")
                gr.Markdown("支持格式：`.txt` `.md` `.docx` `.pdf` `.pptx`")
                gr.Markdown("文档将自动分块 → 向量化 → 存入 Milvus，之后在对话中通过 ReAct 模式提问即可自动检索。")

                file_input = gr.File(
                    label="选择文件",
                    file_types=[".txt", ".md", ".docx", ".pdf", ".pptx"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("上传到知识库", variant="primary")
                upload_result = gr.Textbox(label="上传结果", lines=6, interactive=False)

                upload_btn.click(
                    fn=upload_files,
                    inputs=[file_input],
                    outputs=[upload_result],
                )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Platform Gradio Demo")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="FastAPI 后端地址")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    demo = build_demo(api_url=args.api_url)
    demo.launch(server_port=args.port, share=args.share)
