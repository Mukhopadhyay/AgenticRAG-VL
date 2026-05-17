import chainlit as cl
from scripts.ingest import ingest
from rag.pipeline import RAGPipeline

ingest()


pipeline = RAGPipeline()


@cl.on_message
async def main(message: cl.Message):
    # print(f"Received message: {message}")
    # response = pipeline.run(message.content)
    # response = pipeline.run(message.content)

    msg = cl.Message(content="")

    for chunk in pipeline.stream(message.content):
        msg.content += chunk
        await msg.update()


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass
