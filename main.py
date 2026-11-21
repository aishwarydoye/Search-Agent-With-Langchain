from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
import requests
from langsmith import Client

# Load environment variables
load_dotenv()

# Initialize tools and model
search_tool = TavilySearchResults(max_results=3)
llm = ChatGroq(model="openai/gpt-oss-20b")

# Pull prompt from LangSmith Hub (with fallback)
client = Client()
try:
    prompt = client.pull_prompt(
        "hwchase17/react",
        dangerously_pull_public_prompt=True
    )
    system_prompt = prompt.template
except Exception as e:
    system_prompt = (
        "Answer the following questions as best you can. You have access to the following tools:\n\n"
        "{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )

# Create the agent
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=system_prompt
)

if __name__ == "__main__":
    query = "give current whether of surat"
    print(f"Query: {query}\n")
    response = agent.invoke({'messages': [{'role': 'user', 'content': query}]})
    print("Agent Response:\n")
    print(response["messages"][-1].content)
