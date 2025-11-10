import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tools.sql_agent import sql_graph 
from tools.book_agent import book_graph 
from tools.weather_agent import weather_graph

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
 
# classifies user intent into one of: sql, book, weather
def _coordinator(text: str) -> str:
    t = text.lower() # normalize to lowercase to simplify matching
    # weather keywords
    if any(k in t for k in ["weather", "forecast", "temperature", "rain", "晴れ", "天気", "降水"]):
        return "weather"
    # book keywords
    if any(k in t for k in ["book", "novel", "poem", "quote", "author", "line", "hamlet", "raven", "詩", "小説", "台詞", "作者"]):
        return "book"
    # db keywords 
    if any(k in t for k in ["lego", "titanic", "happiness", "sql", "table", "count", "average", "top", "rank", "group by"]):
        return "sql"
    return ""

ROUTER_SYS = SystemMessage(content=(
    "You are a strict router. Classify the user's last message into exactly one of: "
    "'book', 'weather', 'sql'. Return ONLY the single word."
))

# START: node to route to the appropriate agent
def route_node(state: MessagesState):
    user = state["messages"][-1].content if state.get("messages") else ""
    hint = _coordinator(user)
    if hint:
        return {"route": hint}
    # Use LLM to classify if no hint
    label = llm.invoke([ROUTER_SYS, HumanMessage(content=user)]).content.strip().lower()
    if label not in {"book", "weather", "sql"}:
        label = "sql"  # default to sql if unsure
    return {"route": label}

FINAL_SYS = SystemMessage(content=(
    "You are an expert assistant. "
    "Using the conversation history, produce a final answer to the user's last question. "
    "Incorporate any tool call results as needed. "
    "If you do not have enough information to answer, say so honestly."
)) 

# END: Finalise node: collate tool calls and produce final answer
def finalise(state: MessagesState):
    filtered_msgs = [] 
    tool_call_ids = set() # collect tool call IDs
    for msg in state["messages"]: # for each message
        if hasattr(msg, "tool_calls") and msg.tool_calls: # if it has tool calls
            for tc in msg.tool_calls: # for each tool call
                tool_call_ids.add(tc["id"]) # add its ID to the set
        if hasattr(msg, "tool_call_id") and msg.tool_call_id: # if it has a tool call ID
            tool_call_ids.discard(msg.tool_call_id) # remove it from the set(we have a response for it)
        filtered_msgs.append(msg) # keep the message
    if tool_call_ids: # if there are any remaining tool call IDs
        filtered_msgs = [m for m in filtered_msgs if not (hasattr(m, "tool_call_id") and m.tool_call_id in tool_call_ids)] # remove messages with those IDs
    resp = llm.invoke([FINAL_SYS] + filtered_msgs) # invoke LLM with system message and filtered messages
    return {"messages": [AIMessage(content=resp.content)]} # return final answer message

# Build the state graph
builder = StateGraph(MessagesState)
builder.add_node("route", route_node)
builder.add_node("sql", sql_graph)
builder.add_node("book", book_graph)
builder.add_node("weather", weather_graph)
builder.add_node("finalise", finalise)
builder.add_node("end", finalise)

builder.add_edge(START, "route")

def _on_route(state):
    return state.get("route", "sql")

# Conditional edges from routing node to agents
builder.add_conditional_edges("route", _on_route, {
    "book": "book",
    "sql": "sql",
    "weather": "weather",
})

builder.add_edge("sql", "finalise")
builder.add_edge("book", "finalise")
builder.add_edge("weather", "finalise")
builder.add_edge("finalise", END)
graph = builder.compile()