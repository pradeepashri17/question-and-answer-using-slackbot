import os
import slack
import requests
from flask import Flask
from slackeventsapi import SlackEventAdapter
from pathlib import Path
from dotenv import load_dotenv
from pdf_parser import parse_pdf_pages, store_pages_in_faiss, search_pages
from llm_caller import get_response
import json

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET'], '/slack/events', app)
client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
BOT_ID = client.api_call("auth.test")['user_id']
k_retrieval = 3

processed_events = set()

def download_pdf(file_id):
    file_info = client.files_info(file=file_id).get('file')
    if file_info:
        file_name = file_info['name']
        file_url = file_info['url_private']

        response = requests.get(file_url, headers={'Authorization': f'Bearer {os.environ["SLACK_TOKEN"]}'})
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            return file_name
    return None

@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    event_id = event.get('client_msg_id')  # Get the unique event ID
    user_text = event.get('text', '')  # Get the user's text from the event

    
    if event_id in processed_events:
        return  # Ignore duplicate events
    
    processed_events.add(event_id)  # Add the event ID to the processed set
    
    channel_id = event.get('channel')
    user_id = event.get('user')
    
    if user_id != BOT_ID:
        if 'files' in event:
            for file_info in event['files']:
                if file_info['filetype'] == 'pdf':
                    pdf_file = download_pdf(file_info['id'])
                    if pdf_file:
                        client.chat_postMessage(channel=channel_id, text=f"PDF file '{pdf_file}' downloaded.")
                        parsed_pages = parse_pdf_pages(pdf_file)
                        client.chat_postMessage(channel=channel_id, text=f"PDF parsed.")
                        index, vectorizer = store_pages_in_faiss(parsed_pages)
                        client.chat_postMessage(channel=channel_id, text=f"Stored in FAISS vector DB.")
                        client.chat_postMessage(channel=channel_id, text=f"User query {user_text}")
                        client.chat_postMessage(channel=channel_id, text=f"Fetching LLM Response")

                        query_list = eval(user_text)
                        result_dict = {}

                        for query in query_list:
                            top_similar_pages = search_pages(query, index, vectorizer, parsed_pages)[:k_retrieval]
                            response_dict = get_response(top_similar_pages, query)
                            result_dict.update(response_dict)
                        
                        client.chat_postMessage(channel=channel_id, text="LLM response fetched.")
                        client.chat_postMessage(channel=channel_id, text=str(result_dict))

    

                    else:
                        client.chat_postMessage(channel=channel_id, text="Failed to download the PDF file.")
                    return  # Exit the function after responding to the file upload event
        else:
            client.chat_postMessage(channel=channel_id, text="Please upload a PDF file.")

if __name__ == '__main__':
    app.run(debug=True)