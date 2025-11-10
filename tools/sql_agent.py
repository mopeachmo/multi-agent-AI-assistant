import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

# LLM
CLF_LLM = ChatOpenAI(model="gpt-4.1-mini", temperature=0)       # for routing
SQL_LLM = ChatOpenAI(model="gpt-4.1-mini", temperature=0)       # for SQL agent

# System prompt for the agent
CLASSIFIER_SYS = SystemMessage(content=(
    "You are a strict classifier. "
    "Return EXACTLY one token among: 'titanic', 'happiness', 'lego'. "
    "Pick the most plausible based on wording. Respond with ONLY the one word."
))

# classify titanic or hapinness
def _classify_db(question: str) -> str: # question = user query
    """Return 'titanic', 'happiness' or 'lego'."""
    resp = CLF_LLM.invoke([CLASSIFIER_SYS, HumanMessage(content=question)]) # pass system prompt + user question to LLM
    label = resp.content.strip().lower() # get response text from LLM and normalise 
    print(label)
    if label not in {"titanic", "happiness", "lego"}:
        if "lego" in question.lower():
            label = "lego"
        else:
            label = "happiness"
    return label

# ---- 2) 接続URIは環境変数から取得（ハードコード禁止）----
def _get_db_uri(label: str) -> str:
    env_map = {
        "titanic": "TITANIC_DB_URI",
        "happiness": "HAPPINESS_DB_URI",
        "lego": "LEGO_DB_URI",
    }
    env_key = env_map[label]
    uri = os.getenv(env_key)
    if not uri:
        raise RuntimeError(f"Missing DB URI for '{label}'. Set {env_key} in .env")
    return uri

# load the correct SQLDatabase
def _load_sqldb(which: str) -> SQLDatabase:
    uri = _get_db_uri(which)
    return SQLDatabase.from_uri(uri) # then you can pass the database to SQLDatabaseToolkit

# MAIN: Execute the SQL query via the agent and return the answer
def _run_sql_agent(question: str, db: SQLDatabase) -> str:
    """
    Create a SQL agent and execute the question. Returns the agent's natural-language answer.
    """
    agent = create_sql_agent( # create a LangChain SQL agent
        llm=SQL_LLM,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        top_k=5,  # use top 5 results from DB
        use_query_checker=True, # enable query checking
    )
    result: Dict[str, Any] = agent.invoke({"input": question}) # run the agent with the user question
    out = result.get("output", "").strip()
    if not out:
        out = "I couldn't find an answer to your question."
    return out

# LangGraph Node function
def sql_graph(state) -> Dict[str, Any]:
    """
    Node function for the 'sql' step in coordinator.py.
    Reads the latest user message, routes to the correct DB, runs the SQL agent,
    and returns an AIMessage that the coordinator's 'finalise' node will read.
    """
    # Pull the latest user query (fallback to empty string)
    user_msg = ""
    for m in reversed(state.get("messages", [])): # pick up the latest HumanMessage
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break
        if hasattr(m, "type") and m.type == "human" and hasattr(m, "content"): # support dict-like messages
            user_msg = m.content
            break
    if not user_msg:
        return {"messages": [AIMessage(content="I didn’t receive a question.")]}
    # MAIN LOGIC
    label = _classify_db(user_msg)
    try:
        db = _load_sqldb(label)
        answer = _run_sql_agent(user_msg, db)
        prefix = f"[database: {label}]\n"
        return {"messages": [AIMessage(content=prefix + answer)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"SQL agent error: {e}")]}


# ========= test =========
if __name__ == "__main__":
    test_queries = [
        "How many passengers survived the Titanic disaster?",
        "Which country had the highest happiness score in 2019?",
        "What was the average age of Titanic passengers by class?",
        "Show the top 5 happiest countries overall.",
        "Top five LEGO themes by number of sets.",
        "How many LEGO sets exist in total?"
    ]
    for q in test_queries:
        print("\n=== USER:", q)
        state = {"messages": [HumanMessage(content=q)]}
        out = sql_graph(state)
        print(out["messages"][-1].content)