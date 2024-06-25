from langchain_core.tools import tool
import json
from langchain import hub
from langchain.globals import set_debug, set_verbose
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# set_debug(True)
set_verbose(True)

# from langchain.agents import AgentExecutor, create_tool_calling_agent

# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# prompt.pretty_print()

prompt = ChatPromptTemplate.from_template(
    """
<system-message>
You are a helpful assistant. 
If you are unable to produce answer, just say so.
</system-message>
<intermediate-steps>
{intermediate_steps}
</intermediate-steps>
<human-message>
{input}
</human-message>
<agent-scratchpad>
{agent_scratchpad}
</agent-scratchpad>
"""
)

# prompt = ChatPromptTemplate.from_template(
#     """
# You are a helpful assistant, but you are bad at math.
# Answer user questions using the tools provided below

# {input}

# Scratcpad: {agent_scratchpad}


# """
# )


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    result = first_int * second_int
    print(f"Function: multiply({first_int}, {second_int}) = {result}")
    return result


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    result = first_int + second_int
    print(f"Function: add({first_int}, {second_int}) = {result}")
    return result


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    result = base**exponent
    print(f"Function: exponentiate({base}, {exponent}) = {result}")
    return result


@tool
def hyperspace(from_source: int, to_destination: int) -> int:
    "Travel through hyperspace from one point to another."
    result = from_source + to_destination
    print(f"Function: hyperspace({from_source}, {to_destination}) = {result}")
    return result

# communication tools
contact_book = {}

@tool
def add_contact(contact_name: str, contact_email: str) -> str:
    """Add a contact to the current user contact book. Contact book is a dictionary that associates people names with their email addresses.
    The function stores the email address associated with the provided name in a dictionary called `contact_book`
    """
    contact_book[contact_name] = contact_email
    result = f"Contact {contact_name} added with email {contact_email}"
    print(f"Function: add_contact({contact_name}, {contact_email}) = {result}")
    return result

@tool
def send_email(recipient_address: str, email_subject: str, email_body: str) -> str:
    "Send an email to a recipient with a known e-mail address."
    result = f"Email sent to {recipient_address} with subject {email_subject} and body {email_body}"
    print(f"Function: send_email({recipient_address}, {email_subject}, {email_body}) = {result}")
    return result


@tool
def search_email_address_by_name(name: str) -> str:
    """Search for an email address by name. Use this, if you don't know the email address of a person. 
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
    """
    result = f"Jira ticket created with summary {ticket_summary}, description {ticket_description}, and assignee {assignee_name}"
    print(f"Function: create_jira_ticket({ticket_summary}, {ticket_description}, {assignee_name}) = {result}")
    return result

# math tools
# tools = [multiply, add, exponentiate, hyperspace]


# communication tools
tools = [send_email, create_jira_ticket, search_email_address_by_name, add_contact]

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {
    "max_tokens": 1500,
    "anthropic_version": "bedrock-2023-05-31",
    "stop_sequences": ["User:"],
    "temperature": 1,
    "top_k": 250,
    "top_p": 0.999
}

llm = ChatBedrock(
    model_id=model_id,
    model_kwargs=model_kwargs)

llm_with_tools = llm.bind_tools(tools)

# tools_input = []
# for tool in tools:
#     tools_input.append({
#         "name": tool.name,
#         "description": tool.description,
#         "input_schema": tool.args_schema.schema_json()
#     })
# tools_input = json.dumps(tools_input, indent=4)

# chain = {
#     "input": RunnablePassthrough(),
# } | prompt | llm
# input = "Take 3 to the fifth power as a source point, multiply 5 by two as a destination point, then travel through hyperspace from the source to the destination."
# llm_with_tools.invoke(input)
# chain.invoke(input + "\nAvailable tools: " + tools_input)
# Construct the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    input_str = input("\n\n\nEnter your input: ")
    if input_str == "exit":
        break
    agent_executor.invoke(
        {
            "input": input_str
        }
    )
