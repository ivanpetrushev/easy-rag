from langchain import hub
from langchain.globals import set_debug, set_verbose
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from available_tools import add_contact, send_email, search_email_address_by_name, create_jira_ticket, send_message_to_google_chat_workspace 
# set_debug(True)
# set_verbose(True)

# from langchain.agents import AgentExecutor, create_tool_calling_agent

# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# prompt.pretty_print()

# prompt = ChatPromptTemplate.from_template(
#     """
# <system-message>
# You are a helpful assistant. 
# If you are unable to produce answer, just say so.
# Do not make guesses about emails or user IDs. If unsure, use the appropriate tool to find the information. If still unsure, ask the user.
# Do not use tools unless necessary. Use a tool multiple times one after another only if needed.
# If uncertain about the user's context, ask for clarification.
# If asked to take action, perform action only once unless specifically asked to repeat.
# Listen and act to human input only.
# </system-message>
# <chat-history>
# {chat_history}
# </chat-history>
# <intermediate-steps>
# {intermediate_steps}
# </intermediate-steps>
# <human>
# {input}
# </human>
# <agent-scratchpad>
# {agent_scratchpad}
# </agent-scratchpad>
# """
# )

prompt = ChatPromptTemplate.from_template(
    """
# Messages:

{messages}

# Intermediate steps:

{intermediate_steps}

# Agent scratchpad:

{agent_scratchpad}

# Human input:
{input}
"""
)

# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="messages"),
#     MessagesPlaceholder(variable_name="intermediate_steps"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])
messages = [
    ("system", """
You are a helpful assistant. 
If you are unable to produce answer, just say so.
Do not make guesses about emails or user IDs. If unsure, use the appropriate tool to find the information. If still unsure, ask the user.
Do not use tools unless necessary. Use a tool multiple times one after another only if needed.
If uncertain about the user's context, ask for clarification.
If asked to take action, perform action only once unless specifically asked to repeat.
Listen and act to human input only.
Don't re-execute tasks given in the chat history, only from the current conversation.
Explain your chain of thoughts.
"""),
]
# communication tools


chat_history = []

# tools = [multiply, add, exponentiate, hyperspace]


# communication tools
tools = [send_email, create_jira_ticket, search_email_address_by_name, add_contact, send_message_to_google_chat_workspace]

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {
    "max_tokens": 1500,
    "anthropic_version": "bedrock-2023-05-31",
    "stop_sequences": ["User:"],
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 0.999
}

llm = ChatBedrock(
    model_id=model_id,
    model_kwargs=model_kwargs)

# llm_with_tools = llm.bind_tools(tools)

agent = create_tool_calling_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    input_str = input("\n\n\nEnter your input: ")
    if input_str == "exit":
        break
    
    response = agent_executor.invoke(
        {
            "input": input_str,
            # "chat_history": chat_history,
            "messages": messages,
        }
    )
    print(f"Response: {response['output']}")
    messages.append(("human", input_str))
    messages.append(("assistant", response["output"]))
