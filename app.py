import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langsmith import Client

load_dotenv()

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
        #MainMenu, header, footer { visibility: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 1rem; max-width: 780px; }
        .chat-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .chat-subtitle {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-title">🤖 AI Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Powered by Groq + Tavily Search</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_agent():
    search_tool = TavilySearchResults(max_results=3)
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    try:
        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        prompt = client.pull_prompt(
            "hwchase17/react",
            dangerously_pull_public_prompt=True
        )
        system_prompt = prompt.template
    except Exception:
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

    return create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_prompt
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! Ask me anything — I'll search the web and get you the latest answer. 🌐"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                result = agent.invoke({"messages": history})
                messages_returned = result.get("messages", [])
                reply = messages_returned[-1].content if messages_returned else "Sorry, I couldn't get a response."
            except Exception as e:
                reply = f"Something went wrong: {str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
