import gradio as gr
from dotenv import load_dotenv
import uuid

load_dotenv()

# --- Import Mizan graph ---
from model.graph import mizan_graph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Compile graph with in-memory persistence for demo
memory = MemorySaver()
store = InMemoryStore()
graph = mizan_graph.compile(checkpointer=memory, store=store)


def chat(message, history, user_id, thread_id):
    """Main chat function called by Gradio."""

    if not message.strip():
        return history, history, thread_id

    # Generate thread_id on first message if empty
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Clean user_id
    user_id = user_id.strip() or "anonymous"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    # Stream the graph
    answer = ""
    sources = []

    try:
        for chunk in graph.stream(
            {"query": message, "attempts": 0},
            config=config,
            stream_mode="updates"
        ):
            node_name = next(iter(chunk.keys()))
            node_data = chunk.get(node_name) or {}

            if "answer" in node_data:
                answer = node_data["answer"]
            if "sources" in node_data:
                sources = node_data["sources"]

    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    # Format sources
    sources_text = ""
    if sources:
        valid_sources = [str(s) for s in sources[:3] if s and str(s) != "None"]
        if valid_sources:
            sources_text = "\n\n📎 **Sources:** " + " | ".join(valid_sources)

    full_response = answer + sources_text

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})

    return history, history, thread_id


def clear_chat():
    """Clear chat history and generate new thread."""
    return [], [], str(uuid.uuid4())


# --- Build UI ---
with gr.Blocks(title="Mizan — Egyptian Labor Law Assistant") as demo:

    # State
    thread_id_state = gr.State(str(uuid.uuid4()))

    # Header
    with gr.Column(elem_id="header"):
        gr.HTML("""
            <div style="text-align: center; padding: 20px 0 10px;">
                <h1 style="font-size: 2rem;">⚖️ Mizan</h1>
                <p style="color: #666;">AI-powered Egyptian Labor Law Assistant</p>
            </div>
        """)

    # User ID row
    with gr.Row():
        user_id_input = gr.Textbox(
            label="User ID",
            placeholder="Enter your ID to preserve memory across sessions...",
            value="user_001",
            scale=3
        )
        new_session_btn = gr.Button("🔄 New Session", scale=1, variant="secondary")

    # Chat area
    chatbot = gr.Chatbot(
        label="Conversation",
        elem_id="chatbot",
        show_label=True,
        height=500
    )

    # Input row
    with gr.Row():
        msg_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask about Egyptian labor law... (e.g. What are maternity leave rights?)",
            scale=5,
            lines=2
        )
        submit_btn = gr.Button("Send ⬆️", scale=1, variant="primary")

    # Example questions
    gr.Examples(
        examples=[
            ["What are the rights of a worker regarding maternity leave?"],
            ["How many annual leave days is a worker entitled to?"],
            ["What are the conditions for terminating an employment contract?"],
            ["What are the working hours limits under Egyptian labor law?"],
            ["What compensation is a worker entitled to upon dismissal?"],
        ],
        inputs=msg_input,
        label="Example Questions"
    )

    # History state
    history_state = gr.State([])

    # Wire up events
    submit_btn.click(
        fn=chat,
        inputs=[msg_input, history_state, user_id_input, thread_id_state],
        outputs=[chatbot, history_state, thread_id_state]
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, history_state, user_id_input, thread_id_state],
        outputs=[chatbot, history_state, thread_id_state]
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )

    new_session_btn.click(
        fn=clear_chat,
        outputs=[chatbot, history_state, thread_id_state]
    )

    # Footer
    gr.Markdown("""
    ---
    **Note:** Mizan uses AI to answer questions about Egyptian labor law based on official legal texts.
    This tool is for informational purposes only and does not constitute legal advice.
    Always consult a qualified lawyer for specific legal situations.
    """)


if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
    )