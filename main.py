import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from openai import OpenAI

from constants import GPT_MODEL
from handler import OpenAPIHandler
from models import Model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    app.state.client = client

    assistants = client.beta.assistants.list(limit=100).data

    for assistant in assistants:
        if assistant.name == 'Adjust Advisor':
            app.state.assistant = assistant
            break
    else:
        assistant = client.beta.assistants.create(
            name='Adjust Advisor',
            instructions=(
                'Adjust is an IT-company. You are an customer support expert in various Adjusts product. Use the '
                'below articles from Adjust help center to answer the subsequent question. Search for information in '
                'the articles. If the answer cannot be found in the articles, write "I could not find an answer."'
            ),
            model=GPT_MODEL,
        )
    app.state.assistant = assistant

    yield


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/ask')
async def ask(body: Model, request: Request):
    client = request.app.state.client
    thread = client.beta.threads.create()

    handler = OpenAPIHandler(client, body.question)
    question = handler.make_question()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role='user',
        content=question,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=request.app.state.assistant.id,
    )

    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    answer = [message.content[0].text.value for message in messages.data if message.role == 'assistant'][0]

    return PlainTextResponse(answer)
