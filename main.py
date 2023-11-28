import ast

import pandas as pd
from fastapi import FastAPI

from handler import OpenAPIHandler
from models import Model

app = FastAPI()

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

df = pd.read_csv('./data.csv')
df['embedding'] = df['embedding'].apply(ast.literal_eval)

assistant = client.beta.assistants.create(
    name="Adjust Advisor",
    instructions='Adjust is an IT-company. You are an customer support expert in various Adjusts product. Use the '
                 'below articles from Adjust help center to answer the subsequent question. Search for information in '
                 'the articles. If the answer cannot be found in the articles, write "I could not find an answer."',
    model="gpt-4-1106-preview",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post('/ask')
async def ask(body: Model):
    thread = client.beta.threads.create()
    handler = OpenAPIHandler(body.question)

    message_ = handler.make_question(df)

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    while run.status != 'completed':
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    ans = [message.content[0].text.value for message in messages.data if message.role == 'assistant']

    return ans[0]
