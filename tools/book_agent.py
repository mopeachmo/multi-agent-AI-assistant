import os, json, re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Path to books.json
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "books.json"
BOOKS_PATH = Path(os.getenv("BOOKS_PATH", str(DEFAULT_PATH)))

# to cache loaded books for current session
_BOOKS_CACHE: List[Dict[str, Any]] | None = None

# Load books.json if not already loaded
def _load_books() -> List[Dict[str, Any]]:
    global _BOOKS_CACHE 
    if _BOOKS_CACHE is None:
        if not BOOKS_PATH.exists():
            raise FileNotFoundError(f"[book_agent] books.json not found: {BOOKS_PATH}") 
        with BOOKS_PATH.open("r", encoding="utf-8") as f:
            _BOOKS_CACHE = json.load(f)
        print(f"[book_agent] loaded {len(_BOOKS_CACHE)} records from {BOOKS_PATH}") # for debug
    return _BOOKS_CACHE

# keyword search helpers
_META_FIELDS = ["title", "author", "genre", "chapter", "section", "stanza", "act", "scene", "book"]
_TEXT_FIELDS = ["text", "paragraph", "excerpt"]
_LIST_FIELDS = ["lines"]

# Normalize string (lowercase)
def _norm(s: str) -> str:
    return s.lower()

# Simple tokenizer (split on whitespace and punctuation)
def _tokenise(q: str) -> List[str]:
    return [t for t in re.split(r"[\s、。.,;:!?（）()\\[\\]{}\"'’”“\\-_/]+", q) if t]

# Extract text from relevant fields for scoring
def _field_to_text(rec: Dict[str, Any]) -> str: # rec=1 book record
    chunks = []
    for k in _META_FIELDS + _TEXT_FIELDS:
        v = rec.get(k)
        if isinstance(v, str):
            chunks.append(v)
    for k in _LIST_FIELDS:
        v = rec.get(k)
        if isinstance(v, list):
            chunks.extend([str(x) for x in v]) 
    return _norm(" ".join(chunks)) 

# Score a record based on presence of terms
def _score_record(rec: Dict[str, Any], terms: List[str]) -> int:
    score = 0
    title = _norm(str(rec.get("title", "")))
    author = _norm(str(rec.get("author", "")))
    body = _field_to_text(rec)
    for t in terms:
        if not t:
            continue
        if t in title:  score += 5
        if t in author: score += 4
        if t in body:   score += 2
    return score

# Render a single hit as Markdown
#OUTPUT:  - **The Raven** by Edgar Allan Poe (published 1845) — Poetry
#         › Once upon a midnight dreary... / Over many a quaint and curious volume...
def _render_hit(rec: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"- **{rec.get('title','(no title)')}** by {rec.get('author','(unknown)')}")
    if 'published_year' in rec:
        parts.append(f" (published {rec['published_year']})")
    if 'genre' in rec:
        parts.append(f" — {rec['genre']}")
    snippet = None
    if rec.get("lines"):
        snippet = " / ".join(rec["lines"][:2]) # show up to 2 lines
    snippet = snippet or rec.get("paragraph") or rec.get("text") or rec.get("excerpt")
    if snippet:
        parts.append(f"\n  › {snippet}")
    return "".join(parts)

# MAIN: Search books.json for top 5 matches to query
def _search_books(query: str, k: int = 5) -> List[Dict[str, Any]]: # return dict list
    books = _load_books()
    terms = _tokenise(_norm(query))
    if not terms:
        return []
    scored = [(rec, _score_record(rec, terms)) for rec in books] # list of (rec, score)
    scored = [x for x in scored if x[1] > 0] # remove zero-score
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rec for rec, _ in scored[:k]] # only return top k records

# Finalise as a LangChain tool
@tool("book_search", return_direct=False)
def book_search(query: str, k: int = 5) -> str: # return 5 top matches as markdown
    # docstring
    """ 
    Search the local books.json for relevant entries.
    Args:
        query: user question or keywords
        k: number of top hits
    Returns:
        Markdown summary of top matches (title/author/year/genre/snippet)
    """
    print("[book_agent] book_search CALLED with:", query) # for debug
    hits = _search_books(query, k=k) 
    if not hits:
        return "No match. Try specifying a title, author, or a distinctive quote."
    body = "\n".join(_render_hit(h) for h in hits) # render as markdown
    return f"Top matches from books.json:\n{body}" # return markdown
# OUTPUT: Top matches from books.json:
#         - **The Raven** by Edgar Allan Poe (published 1845) — Poetry
#         › Once upon a midnight dreary... / Over many a quaint and curious volume...

# To pass to the agent
tools = [book_search] 

# System prompt for the agent
SYS = """You are a helpful book-RAG agent.
- Only answer if the query is about literature (books, poems, authors, quotes, lines, plots).
- If it looks like data/analytics or databases (e.g., LEGO, Titanic, happiness rankings), reply with: "NOT_RELEVANT".
- Always call `book_search` for book queries and answer ONLY from its results.
"""

# Bind LLM with tools 
llm_with_tools = llm.bind_tools(tools)

# Agent node function
def _agent_node(state: MessagesState) -> Dict[str, Any]:
    msgs = [SystemMessage(content=SYS)] + state["messages"] # prepend system prompt + user msgs
    ai = llm_with_tools.invoke(msgs) 
    return {"messages": [ai]}

# ToolNode for the tools
tool_node = ToolNode(tools)

# Build the StateGraph
graph = StateGraph(MessagesState)
graph.add_node("agent", _agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)   # ツール呼ぶなら tools へ
graph.add_edge("tools", "agent")                        # 1 回だけ戻って最終回答
graph.add_edge("agent", END)

book_graph = graph.compile()

# ========= test =========
if __name__ == "__main__":
    test_queries = [
        "Who wrote The Raven? Give one famous line.",
        "Show me about Moby Dick.",
        "Show me lines about 'two roads diverged'.",
    ]
    for q in test_queries:
        print("\n=== USER:", q)
        state = {"messages": [HumanMessage(content=q)]}
        out = book_graph.invoke(state)
        print(out["messages"][-1].content)