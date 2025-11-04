import os, json, requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# MAIN: Weather tool
# OUTPUT: {"status":"ok","city":"London","weather":"Partly cloudy","temperature_c":14.0}
@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a given city using WeatherAPI.com.
    Args:
        city (str): City name, e.g., "London".
    Returns:
        str: JSON string with weather description and temperature in Celsius, or an error message.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({"status":"error","message":"API key not set"}, ensure_ascii=False)
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&lang=en"
        r = requests.get(url, timeout=10) # 10 seconds timeout to avoid hanging
        r.raise_for_status() # raise error for HTTP status 4xx/5xx -> except
        data = r.json()
        desc = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        return json.dumps({
            "status": "ok",
            "city": city,
            "weather": desc,
            "temperature_c": temp
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# city decision node
def decide_city(state: MessagesState):
    user = state["messages"][-1].content if state["messages"] else "" # get latest user message or blank ""
    sys = {
        "role": "system",
        "content": (
            "You are a weather assistant. Extract the city name from the user text. "
            "If no clear city is present, ask a short follow-up question to clarify. "
            "When you know the city, call the tool get_weather with {city}."
        ),
    }
    resp = llm.bind_tools([get_weather]).invoke([sys] + state["messages"]) # pass system + user msgs to LLM 
    return {"messages": [resp]}

builder = StateGraph(MessagesState)
builder.add_node("decide_city", decide_city)
builder.add_node("tools", ToolNode([get_weather]))  # 天気ツールはこのサブグラフ内でだけ使う

builder.add_edge(START, "decide_city")
builder.add_conditional_edges("decide_city", tools_condition)  # ツール呼び出しがあれば tools へ
builder.add_edge("tools", END)

weather_graph = builder.compile()


# ========= test =========
if __name__ == "__main__":
    test_queries = [
        "What's the weather in Tokyo?",
        "How is it in London right now?",
        "天気教えて（都市不明）",
        "Paris weather please",
    ]
    for q in test_queries:
        print("\n=== USER:", q)
        state = {"messages": [HumanMessage(content=q)]}
        out = weather_graph.invoke(state)
        print(out["messages"][-1].content)