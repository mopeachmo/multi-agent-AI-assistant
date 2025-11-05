import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

load_dotenv()

# to make sure we can find the database
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # this directory
DATA_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "data")) # data directory
TITANIC_DB_PATH = os.path.join(DATA_DIR, "titanic.db")
HAPPY_DB_PATH = os.path.join(DATA_DIR, "happiness_index.db")

# LLM
CLF_LLM = ChatOpenAI(model="gpt-4.1-mini", temperature=0)       # for routing
SQL_LLM = ChatOpenAI(model="gpt-4.1-mini", temperature=0)       # for SQL agent

# System prompt for the agent
CLASSIFIER_SYS = SystemMessage(content=(
    "You are a strict classifier. Given a user question, answer exactly one token: "
    "'titanic' if it refers to Titanic passengers/ship data; "
    "'happiness' if it refers to World Happiness/Year/Rank/Score style data. "
    "If uncertain, choose the most likely based on wording. Respond with ONLY the one word."
))

# classify titanic or hapinness
def _classify_db(question: str) -> str: # question = user query
    """Return 'titanic' or 'happiness'."""
    resp = CLF_LLM.invoke([CLASSIFIER_SYS, HumanMessage(content=question)]) # pass system prompt + user question to LLM
    label = resp.content.strip().lower() # get response text from LLM and normalise 
    print(label)
    return "titanic" if "titanic" in label else "happiness"

# load the correct SQLDatabase
def _load_sqldb(which: str) -> SQLDatabase:
    if which == "titanic": # which = "titanic" or "happiness"
        path = TITANIC_DB_PATH
    else:
        path = HAPPY_DB_PATH
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SQLite file not found at {path}. Make sure your DBs are in a folder named 'data'."
        )
    uri = f"sqlite:///{path.replace(os.sep, '/')}"
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
        verbose=False
    )
    result: Dict[str, Any] = agent.invoke({"input": question}) # run the agent with the user question
    return result.get("output", "").strip() # return the answer text in natural language

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
    which = _classify_db(user_msg)
    try:
        db = _load_sqldb(which)
        answer = _run_sql_agent(user_msg, db)
    except Exception as e:
        # Return error text as an AIMessage so finalise can still compose
        return {"messages": [AIMessage(content=f"SQL agent error: {e}")]}
    # Helpful tag to indicate which DB was used (kept concise for finalise)
    prefix = f"[database: {which}]\n"
    return {"messages": [AIMessage(content=prefix + answer)]}


# ========= test =========
if __name__ == "__main__":
    test_queries = [
        "How many passengers survived the Titanic disaster?",
        "Which country had the highest happiness score in 2019?",
        "What was the average age of Titanic passengers by class?",
        "Show the top 5 happiest countries overall.",
    ]
    for q in test_queries:
        print("\n=== USER:", q)
        state = {"messages": [HumanMessage(content=q)]}
        out = sql_graph(state)
        print(out["messages"][-1].content)