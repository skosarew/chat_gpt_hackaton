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

    yield


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/ask')
async def ask(body: Model, request: Request):
    client = request.app.state.client

    handler = OpenAPIHandler(client, body.question)
    question = handler.make_question()

    messages = [
        {
            'role': 'system',
            'content': (
                'Adjust is an IT-company. You are an customer support expert in various Adjusts product. Use the '
                'below articles from Adjust help center to answer the subsequent question. Search for information in '
                'the articles. If the answer cannot be found in the articles, write "I could not find an answer."'
            )
        },
        {
            'role': 'user',
            'content': question
        }
    ]

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )

    answer = completion.choices[0].message.content

    return PlainTextResponse(answer)
