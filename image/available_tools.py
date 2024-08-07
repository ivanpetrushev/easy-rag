import os
import json
import requests
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings


contact_book = {}

@tool
def add_contact(contact_name: str, contact_email: str) -> str:
    """Add a contact to the current user contact book. Contact book is a dictionary that associates people names with their email addresses.
    The tool should be used only once per contact and only if it is new. If the contact already exists, the tool should not be used.
    """
    contact_book[contact_name] = contact_email
    result = f"Contact {contact_name} added with email {contact_email}"
    print(f"Function: add_contact({contact_name}, {contact_email}) = {result}")
    return result

@tool
def send_email(recipient_address: str, email_subject: str, email_body: str) -> str:
    "Send an email to a recipient with a known e-mail address. This tool should be used only once per email."
    result = f"Email sent to {recipient_address}\n Subject: {email_subject}\n Body: {email_body}"
    print(f"Function: send_email({recipient_address}, {email_subject}, {email_body}) = {result}")
    return result


@tool
def search_email_address_by_name(name: str) -> str:
    """Search for an email address in current user contact book. Use this, if you need the email address of a person and you don't know it. 
    If the email address is not found, the function returns 'Email address not found'.
    In this case it is not useful to invoke the same function with the same name again.
    """
    result = contact_book.get(name, "Email address not found")
    print(f"Function: search_email_address_by_name({name}) = {result}")
    return result


@tool
def create_jira_ticket(ticket_summary: str, ticket_description: str, assignee_name: str) -> str:
    """
    Create a Jira ticket with a summary, description, and assignee. 
    Jira tickets are instructions for tasks given to a specific team member.
    Target team member must be specified with his/her name.
    Don't create multiple Jira tickets for the same task.
    """
    result = f"Jira ticket created with summary {ticket_summary}, description {ticket_description}, and assignee {assignee_name}"
    print(f"Function: create_jira_ticket({ticket_summary}, {ticket_description}, {assignee_name}) = {result}")
    return result

@tool
def send_message_to_google_chat_workspace(workspace_name: str, message: str):
    """
    Send a message to a Google Chat workspace. Don't use this to send repetetive messages.
    """
    print(f"Function: send_message_to_google_chat_workspace({workspace_name}, {message})")
    workspace_webhooks = {
        "test_bot_space": os.environ.get("GCHAT_TEST_BOT_SPACE_WEBHOOk"),
    }
    if workspace_name not in workspace_webhooks:
        return f"Webhook for workspace not found: {workspace_name}"
    
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
    }
    data = {
        "text": message,
    }
    response = requests.post(workspace_webhooks[workspace_name], headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        return f"Failed to send message to Google Chat workspace {workspace_name}: {response.text}"
    
    result = f"Message sent to Google Chat workspace {workspace_name}: {message}"
    
    return result


def get_embeddings():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    return embeddings

db = None
def load_chromadb():
    global db
    if db is not None:
        return db
    # load ChromaDB only if /chromadb is empty
    if os.listdir("/chromadb"):
        # use ChromaDB from disk
        tmp_db = Chroma(persist_directory="/chromadb", embedding_function=get_embeddings())
        db = tmp_db
    else:
        print(f"ChromaDB data not found in /chromadb")
    
@tool
def t60_t14_p50_retriever(query: str) -> str:
    """
    Retrieve information from the P50, T60 or T14 database based on the query.
    Usage:
    - Use this when asked technical information about P50, T60 and T14.
    - Pass the whole user input as `query` parameter.
    Output format instructions:
    1. Start with the direct answer to the query without any preamble.
    2. After the answer, provide a list of relevant documents in the following format:

    Relevant Documents:
    - Source: [filename/url], Page: [page number], Relevance: [0-100]
    - Source: [filename/url], Page: [page number], Relevance: [0-100]
    (Include as many documents as you use to compose the answer. "Relevance" is a score determined by you)
    
    Note: Always include the "Relevant Documents" section.
    """
    print(f"Function: t60_t14_retriever({query})")
    load_chromadb()
    if db is None:
        return "ChromaDB not loaded"
    results = db.similarity_search(query)

    # debug - print all results in order (with index) with metadata and page_content
    for i, result in enumerate(results):
        print(f"\n===============> Result {i}: {result.metadata}")
        print(f"Page content: {result.page_content}")
    
    return results

@tool
def get_weather(location: str):
    """
    Get the weather for a given location. 
    Fetches a weater API that responds with JSON. 
    Answer only in metric units, unless instructed otherwise.
    """
    print(f"Function: get_weather({location})")
    url = f"http://api.weatherapi.com/v1/current.json?key={os.environ['WEATHER_API_KEY']}&q={location}&aqi=no"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Failed to get weather for {location}"
    
    return response.text