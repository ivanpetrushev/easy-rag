from langchain import hub
from langchain.globals import set_debug, set_verbose
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from available_tools import add_contact, send_email, search_email_address_by_name, create_jira_ticket, send_message_to_google_chat_workspace 
# set_debug(True)
set_verbose(True)

# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# prompt = hub.pull("hwchase17/react")
# prompt.pretty_print()


system_msg = """
You are a helpful assistant. 
If you are unable to produce answer, just say so.
Do not make guesses about emails or user IDs. If unsure, use the appropriate tool to find the information. If still unsure, ask the user.
Do not use tools unless necessary. Use a tool multiple times one after another only if needed.
If uncertain about the user's context, ask for clarification.
If asked to take action, perform action only once unless specifically asked to repeat.
Please only respond to and act on the most recent human query. Previous messages are provided for context only.
Explain your chain of thoughts.
"""

# order is very important, or we get errors regarding "alternating roles" and "assistant cant answer last when using tools"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


memory = ChatMessageHistory(session_id="test-session")

# communication tools
tools = [send_email, create_jira_ticket, search_email_address_by_name, 
         add_contact, send_message_to_google_chat_workspace]

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

agent = create_tool_calling_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor, lambda session_id: memory, input_messages_key="input", history_messages_key="chat_history")

while True:
    input_str = input("\n\n\nEnter your input: ")
    if input_str == "exit":
        break

    response = agent_with_chat_history.invoke(
        {
            "input": input_str,
        },
        config={
            "configurable": {
                "session_id": "test-session"
            }
        }
    )
