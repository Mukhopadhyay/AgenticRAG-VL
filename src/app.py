import chainlit as cl
from scripts.ingest import ingest
from rag.pipeline import RAGPipeline

ingest()

pipeline = RAGPipeline()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    async for chunk in pipeline.astream(message.content):
        await msg.stream_token(chunk)


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass
