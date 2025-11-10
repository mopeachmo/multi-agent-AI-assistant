# Multi-Agent AI Assistant

## Description
**Multi-Agent AI Assistant** is a Streamlit-based conversational application that integrates **three specialised AI agents** – a **Book Agent**, a **Weather Agent**, and a **SQL Agent** – orchestrated through a **Coordinator** to produce unified, context-aware responses.

The goal of this project was to explore how multiple domain-specific agents can collaborate through an LLM-driven coordination layer. It was built as part of an internship focused on **AI-enabled digital solutions** and demonstrates skills in:
- multi-agent system design using **LangGraph / LangChain**
- integration of external APIs and local data sources
- user interface design and deployment via **Streamlit**

This project solves the challenge of managing heterogeneous data queries in one interface. Users can ask about books, data analytics, or the weather in natural language, and the appropriate agent handles the request before the Coordinator composes the final answer.

---

## Features

### Integrated AI Agents
- **Book Agent**  
  Retrieves literary information by searching a local `books.json` file for authors, quotes, genres, and themes.  
  → Example: *“Who wrote The Raven?”* or *“What’s a famous line from Hamlet?”*

- **Weather Agent**  
  Fetches real-time weather information via the [WeatherAPI](https://www.weatherapi.com/).  
  It can even infer or clarify the city name if you’re unsure.  
  → Example: *“What’s the weather like in Edinburgh today?”*

- **SQL Agent**  
  Answers analytical questions using three PostgreSQL databases:  
  `lego` (Lego parts and sets data) `titanic` (Titanic passenger data) and `happiness_index` (World Happiness data).  
  → Example: *“Top five LEGO themes by number of sets.” / “How many Titanic survivors?”*

### Coordinator
The Coordinator routes each question to the relevant agent and refines their responses into a coherent final answer.

### Streamlit UI
- Clean interface with icons and sub-titles for each agent.  
- User-friendly chat window for free-form questions.  

---

## How to Use

### 0️⃣ Prerequisites
- Docker Desktop + Docker Compose v2
- An OpenAI API key; a WeatherAPI key

### 1️⃣ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-ai-assistant.git
cd multi-agent-ai-assistant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # (on Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Set up environment variables
```bash
# Create a .env file in the project root and include:
OPENAI_API_KEY=your_openai_key
WEATHER_API_KEY=your_weatherapi_key
```

### 3️⃣ Build & run
```bash
docker compose build --no-cache
docker compose up -d
# Open your browser at http://localhost:8501
```
