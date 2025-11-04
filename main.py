import os
import json
import streamlit as st
from dotenv import load_dotenv
from coordinator import graph  

load_dotenv()

# Title and description
st.title("📚🌦️🗃️ Multi-Agent AI Assistant")
st.markdown(
    """
### 🤖 Integrated AI Agents
This application brings together **three specialised agents** working collaboratively:

- **📚 Book Agent** – Answers questions about literature by searching a local `books.json` file (titles, authors, quotes, and themes).  
- **🌦️ Weather Agent** – Retrieves current weather data for cities worldwide using *WeatherAPI.com*. If you're unsure of a city name, this agent can help you find it.   
- **🗃️ Database Agent** – Interprets analytical questions and queries local SQLite databases (`titanic.db` and `happiness_index.db`) to produce natural-language summaries.  

Each agent contributes evidence or analysis which the system’s *Coordinator* then refines into a final response.
""",
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = {}

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if i in st.session_state.tool_calls:
            t = st.session_state.tool_calls[i]
            st.info(
                f"**Tool Executed:** `{t.get('name','')}`\n\n"
                f"**Input:**\n```json\n{t.get('args', {})}\n```\n"
                f"**Output:**\n```json\n{t.get('result', {})}\n```"
            )

if prompt := st.chat_input("Ask something?"):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    steps = graph.stream({"messages": st.session_state["messages"]}, stream_mode="values")
    final_reply = None
    tool_info_for_this_turn = None
    for step in steps:
        msgs = step.get("messages", [])
        if not msgs:
            continue
        last = msgs[-1]
        mtype = getattr(last, "type", None)
        if mtype == "tool":
            tool_info_for_this_turn = {
                "name": getattr(last, "name", ""),
                "args": getattr(last, "input", {}),
                "result": getattr(last, "content", {}),
            }
        if mtype == "ai":
            final_reply = last.content
    if final_reply is None:
        final_reply = "Done."
    st.session_state["messages"].append({"role": "assistant", "content": final_reply})
    with st.chat_message("assistant"):
        st.markdown(final_reply)
        if tool_info_for_this_turn:
            st.info(
                f"**Tool Executed:** `{tool_info_for_this_turn.get('name','')}`\n\n"
                f"**Input:**\n```json\n{tool_info_for_this_turn.get('args', {})}\n```\n"
                f"**Output:**\n```json\n{tool_info_for_this_turn.get('result', {})}\n```"
            )
    if tool_info_for_this_turn:
        idx = len(st.session_state["messages"]) - 1
        st.session_state["tool_calls"][idx] = tool_info_for_this_turn