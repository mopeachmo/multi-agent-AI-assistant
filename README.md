# Multi-Agent AI Assistant  
**LLM-Orchestrated Domain-Specialised Agent System**

---


## Overview

This project is a **multi-agent conversational AI system** built with **LangGraph / LangChain**, **PostgreSQL**, and **Streamlit**.

The goal of this project was to design a system where multiple specialised AI agents can work together under a coordination layer. Instead of building one large assistant that tries to do everything, I created separate agents, each responsible for a specific domain. A central Coordinator decides which agent should handle the user’s question and then combines the results into a clear response.

This project demonstrates practical skills in:

- Multi-agent system design  
- LLM orchestration and routing  
- API and database integration  
- Containerised deployment using Docker  
- Building interactive web apps with Streamlit  

---

## System Architecture
```bash
User Query
↓
Coordinator (LLM Router + Response Synthesiser)
↓
┌──────────────┬──────────────┬──────────────┐
│ Book Agent │ Weather Agent│ SQL Agent │
└──────────────┴──────────────┴──────────────┘
```
---


### Design Principles

- **Modular design** – Each agent has a clear responsibility.
- **Tool-based reasoning** – Agents use structured tools such as SQL queries or API calls.
- **Extensibility** – New agents can be added without changing the whole system.
- **Grounded responses** – Answers are based on structured data sources.

---

## Agents

### 1. Book Agent

The Book Agent searches a local `books.json` file.

**Functions**
- Find authors  
- Retrieve quotes  
- Search by genre or theme  

**Example questions**
- “Who wrote *The Raven*?”
- “What is a famous line from *Hamlet*?”

**Implementation details**
- JSON data parsing  
- Retrieval-based answering  
- Dataset-grounded responses  

---

### 2. Weather Agent

The Weather Agent retrieves real-time weather data using WeatherAPI.

**Functions**
- Get current weather conditions  
- Handle unclear city names  

**Example question**
- “What’s the weather like in Edinburgh today?”

**Implementation details**
- REST API integration  
- Secure key management via `.env` file  
- Live data fetching  

---

### 3. SQL Agent

The SQL Agent answers analytical questions using three PostgreSQL databases:

- `lego` – LEGO sets and themes  
- `titanic` – Titanic passenger data  
- `happiness_index` – World Happiness data  

**Example questions**
- “Top five LEGO themes by number of sets.”
- “How many Titanic survivors?”
- “Which country ranks highest in happiness?”

**Implementation details**
- LLM-generated SQL queries  
- Schema-aware query generation  
- Docker-based PostgreSQL services  
- Structured result formatting  

---

## Coordinator

The Coordinator manages the system.

**Main responsibilities**
- Understand user intent  
- Route the query to the correct agent  
- Collect outputs  
- Generate a final clear answer  

This component shows my understanding of multi-agent workflows and LLM-based routing logic.

---

## User Interface

The application is built with **Streamlit**.

**Features**
- Chat-style interface  
- Clear layout  
- Easy local deployment  
- Docker-compatible setup  

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph / LangChain |
| LLM | OpenAI API |
| Backend Data | PostgreSQL |
| External API | WeatherAPI |
| Frontend | Streamlit |
| Deployment | Docker |
| Language | Python |

---

## How to Use

### 0️⃣ Prerequisites
- Docker Desktop + Docker Compose v2
- An OpenAI API key; a WeatherAPI key

### 1️⃣ Installation
```bash
# Clone the repository
git clone https://github.com/mopeachmo/multi-agent-ai-assistant.git
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

---

## Key Learning Outcomes
Through this project, I developed skills in:
- Designing modular AI systems
- Integrating structured and external data sources
- Implementing LLM-to-SQL workflows
- Managing containerised services
- Building practical AI applications

---

## Future Improvements
- Add conversation memory layer
- Improve error handling
- Introduce user authentication
- Deploy to a cloud platform

---

## Summary
This project shows my ability to design and implement a multi-agent AI system from end to end. It combines LLM orchestration, database integration, API usage, and frontend development into one structured and extensible application.