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

# sys message for final answer composition
FINAL_SYS = SystemMessage(content="You are the final answer composer. Read the prior messages and produce a concise and helpful answer. DO NOT over write the prior answers! Just summrise or make it easier to read. ")

# The coordinator node: classifies user intent into one of: database, book, weather.
def coordinator(state):
    user = state["messages"][-1].content if state["messages"] else "" # get latest user message
    label = llm.invoke( 
        f"Classify intent: one of [database, book, weather]. User: {user}\nAnswer one word only."
    ).content.strip().lower() # remove whitespace and make lowercase
    return {"intent": label}

# Finalise node: collate tool calls and produce final answer
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
builder.add_node("coordinator", coordinator)
builder.add_node("database", sql_graph)
builder.add_node("book", book_graph)
builder.add_node("weather", weather_graph)
builder.add_node("finalise", finalise)
builder.add_node("end", finalise)

builder.add_edge(START, "coordinator")
builder.add_conditional_edges(
    "coordinator",
    lambda s: s.get("intent", "end")
)
builder.add_edge("database", "finalise")
builder.add_edge("book", "finalise")
builder.add_edge("weather", "finalise")
builder.add_edge("finalise", END)
graph = builder.compile()