# AI Agent Journey 🤖

Learning to build AI Agents and automation tools — day by day.
Building toward a career as an AI Engineer in Europe.

## 🛠 Tech Stack

`Python` `asyncio` `aiohttp` `requests` `python-dotenv` `FastAPI` `uvicorn` `OpenAI SDK` `GapGPT API` `n8n` `Telegram Bot API` `ChromaDB` `sentence-transformers` `LangChain` `LangGraph`

## 📂 Structure

| Folder | Contents |
|--------|----------|
| `week1/` | Python foundations, async, APIs, error handling |
| `week2/` | Professional Git workflow notes |
| `week3/` | API auth, LLM client, FastAPI webhook |
| `week4/` | Tool use, memory, AI assistant CLI |
| `phase2_week1/` | n8n workflows and automation |
| `phase2_week2/` | RAG pipeline and vector database |
| `phase2_week3/` | LangChain and LangGraph |
| `n8n-workflows/` | Exported n8n workflow JSON files |

## 📁 Projects

### Week 1 — Python Foundations
| File | Description |
|------|-------------|
| `week1/day2_api_basics.py` | Basic GET/POST requests with JSON parsing |
| `week1/day3_env_secrets.py` | Secure API key management with .env |
| `week1/day4_error_handling.py` | Error handling and logging for API calls |
| `week1/day5_async_basics.py` | Sync vs Async performance comparison |
| `week1/week1_project.py` | Async multi-API fetcher with full error handling |

### Week 2 — Professional Git
| File | Description |
|------|-------------|
| `week2/week2_notes.md` | Git branching, stash, conflict resolution notes |

### Week 3 — API Auth & LLM Integration
| File | Description |
|------|-------------|
| `week3/week3_day1_http.py` | HTTP headers, query params, and pagination |
| `week3/week3_day2_auth.py` | Authentication types and LLM API integration |
| `week3/week3_day3_retry.py` | Rate limit handling with exponential backoff |
| `week3/week3_day4_webhook.py` | FastAPI webhook with AI response handler |
| `week3/week3_project.py` | Multi-provider LLM client with fallback |

### Week 4 — LLM Tool Use & First AI Agent
| File | Description |
|------|-------------|
| `week4/week4_day1_tools.py` | LLM tool use with ReAct agent loop |
| `week4/week4_day2_multi_tool.py` | Multi-tool agent with system prompt |
| `week4/week4_day3_memory.py` | Agent with conversation memory |
| `week4/week4_day4_prompts.py` | Prompt engineering with structured output |
| `week4/week4_day5_error_handling.py` | Safe agent with error handling |
| `week4/week4_project.py` | Complete AI assistant CLI |

### Phase 2 Week 1 — n8n Fundamentals
| Workflow | Description |
|----------|-------------|
| `n8n-workflows/phase2_week1_first_workflow.json` | First n8n workflow |
| `n8n-workflows/phase2_week1_error_handling.json` | Error handling with IF node |
| `n8n-workflows/phase2_week1_code_node.json` | Data processing with Code node |
| `n8n-workflows/phase2_week1_telegram.json` | Telegram integration |
| `n8n-workflows/phase2_week1_telegram_bot.json` | Telegram bot with webhook |
| `n8n-workflows/phase2_week1_news_digest.json` | Daily news digest with GapGPT |

### Phase 2 Week 2 — RAG & Vector Database
| File | Description |
|------|-------------|
| `phase2_week2/day1_embeddings.py` | Text embeddings and semantic search |
| `phase2_week2/day2_chroma.py` | Persistent vector storage with ChromaDB |
| `phase2_week2/day3_rag_pipeline.py` | Complete RAG pipeline |
| `phase2_week2/day4_chunking.py` | Document chunking strategies |
| `phase2_week2/day5_advanced_rag.py` | Advanced RAG with metadata filtering |
| `phase2_week2/day6_project_qa_bot.py` | Customer support Q&A bot |

### Phase 2 Week 3 — LangChain & LangGraph
| File | Description |
|------|-------------|
| `phase2_week3/day1_langchain_basics.py` | ChatOpenAI, prompt templates, chain operator |
| `phase2_week3/day2_langchain_tools.py` | @tool decorator and create_react_agent |
| `phase2_week3/day3_langchain_memory.py` | Session memory with RunnableWithMessageHistory |
| `phase2_week3/day4_langchain_rag.py` | RAG as an agent tool with ChromaDB |
| `phase2_week3/day5_langgraph_persistence.py` | LangGraph MemorySaver with thread isolation |
| `phase2_week3/day6_project_langchain_agent.py` | Complete agent: RAG + tools + LangGraph memory |

## 🚀 Setup

```bash
git clone https://github.com/sina-naderi/ai-agent-journey.git
cd ai-agent-journey
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GAP_API_KEY=your_gapgpt_key_here
```

## 🤖 Run Projects

```bash
# Phase 1 AI Assistant
python week4/week4_project.py

# RAG Q&A Bot
python phase2_week2/day6_project_qa_bot.py

# LangChain Agent
python phase2_week3/day6_project_langchain_agent.py
```

## 📈 Progress

- [x] Week 1 — Python Foundations
- [x] Week 2 — Professional Git
- [x] Week 3 — API Auth & LLM Integration
- [x] Week 4 — LLM Tool Use & First AI Agent
- [x] Phase 2 Week 1 — n8n Fundamentals
- [x] Phase 2 Week 2 — RAG & Vector Database
- [x] Phase 2 Week 3 — LangChain & LangGraph
- [ ] Phase 2 Week 4 — PostgreSQL, FastAPI & Capstone Agent
- [ ] Phase 3 — Portfolio Projects & Deployment
- [ ] Phase 4 — Career Stack & First Applications
- [ ] Phase 5 — Interview Prep & System Design
