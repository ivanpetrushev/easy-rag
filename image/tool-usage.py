from langchain_core.tools import tool

from langchain import hub
from langchain.globals import set_debug
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts.chat import ChatPromptTemplate


set_debug(True)

# from langchain.agents import AgentExecutor, create_tool_calling_agent

# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# prompt.pretty_print()

prompt = ChatPromptTemplate.from_template(
    """
================================ System Message ================================

You are a helpful assistant


================================ Human Message =================================

{input}

============================= Messages Placeholder =============================

{agent_scratchpad}


"""
)


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


tools = [multiply, add, exponentiate, hyperspace]

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
llm_with_tools.invoke("Take 3 to the fifth power as a source point, multiply 5 by two as a destination point, then travel through hyperspace from the source to the destination.")

# Construct the tool calling agent
# agent = create_tool_calling_agent(llm, tools, prompt)

# # Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke(
#     {
#         "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
#     }
# )
