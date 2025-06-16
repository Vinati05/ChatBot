from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs_parsers import PydanticOutputParser
from tools import search_tool , wikipedia_tool ,save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

print("API KEY USED:", os.getenv("OPENROUTER_API_KEY"))  # Debug: show which key is loaded

llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",  # Switch to a different model on OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)


parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the ouput in this formal and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        (placeholder, "{agent_scratchpad}"),
    ]
).partial(
    format_instructions=parser.get_format_instructions(),
)

tools = [search_tool , wikipedia_tool , save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # Enable verbose output to see the thought process
)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})  # Return raw response for debugging purposes

structured_response = parser.parse(raw_response.get("output")[0]["text"])
print("STRUCTURED RESPONSE:", structured_response.topic)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
except Exception as e:
    print("Error parsing response:", e, "Raw response:", raw_response)
