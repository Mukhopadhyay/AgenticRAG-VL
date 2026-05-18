import chainlit as cl
from scripts.ingest import ingest
from graph.workflow import workflow

ingest()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    initial_state = {
        "query": message.content,
        "rewritten_query": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "answer": "",
        "verified": False,
        "retry_count": 0,
    }

    final_state = await workflow.ainvoke(initial_state)
    answer = final_state.get("answer") or "I could not find an answer to your question."
    await msg.stream_token(answer)


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass
