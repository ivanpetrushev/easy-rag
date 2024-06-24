from langchain_core.tools import tool
import json
from langchain import hub
from langchain.globals import set_debug
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

set_debug(True)

# from langchain.agents import AgentExecutor, create_tool_calling_agent

# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()

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


@tool
def send_email(recipient_address: str, subject: str, body: str) -> str:
    "Send an email to a recipient."
    result = f"Email sent to {recipient_address} with subject {subject} and body {body}"
    print(f"Function: send_email({recipient_address}, {subject}, {body}) = {result}")
    return result


@tool
def search_email_address_by_name(name: str) -> str:
    "Search for an email address by name. Use this, if you don't know the email address of a recipient."
    email_map = {
        "Alice": "alice@dot.com",
        "Bob": "bob@dot.com"
    }
    result = email_map.get(name, "Email address not found")
    print(f"Function: search_email_address_by_name({name}) = {result}")
    return result


@tool
def create_jira_ticket(summary: str, description: str, assignee: str) -> str:
    "Create a Jira ticket with a summary, description, and assignee."
    result = f"Jira ticket created with summary {summary}, description {description}, and assignee {assignee}"
    print(f"Function: create_jira_ticket({summary}, {description}, {assignee}) = {result}")
    return result

# math tools
# tools = [multiply, add, exponentiate, hyperspace]


# communication tools
tools = [send_email, create_jira_ticket, search_email_address_by_name]

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
    input_str = input("Enter your input: ")
    if input_str == "exit":
        break
    agent_executor.invoke(
        {
            "input": input_str
        }
    )
