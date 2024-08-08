import os
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Set up the OpenAI API key
openai_api_key = 'openai-api-key'

# Create an instance of the OpenAI class
client = OpenAI(api_key=openai_api_key)

# Define the prompt templates
SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions given context.
Context is provided to you within the <context> </context> tags, and the list of 
questions are provided to you within the <questions> </questions> tag.
The output should be in the form JSON blob that pairs each question with its corresponding answer.
Answers should be word-to-word match if the question is a word-to-word match.
If the answer is low confidence, reply with "Data Not Available".
"""

USER_PROMPT = """
<context>
{context}
</context>

<questions>
{questions}
</questions>

Strictly follow the following output format:
{{
  "question_1": "answer_1",
  "question_2": "answer_2",
  ...
}}
"""

def get_response(context: str, questions: str) -> dict:
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context, questions=questions)}
        ]
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        result = {}

    return result