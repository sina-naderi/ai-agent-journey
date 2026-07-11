# AI Agent Journey 🤖

Learning to build AI Agents and automation tools — day by day.
Building toward a career as an AI Engineer in Europe.

## 🛠 Tech Stack

`Python` `asyncio` `aiohttp` `requests` `python-dotenv` `FastAPI` `uvicorn` `OpenAI SDK` `GapGPT API` `n8n` `Telegram Bot API` `ChromaDB` `sentence-transformers`

## 📂 Structure

| Folder | Contents |
|--------|----------|
| `week1/` | Python foundations, async, APIs, error handling |
| `week2/` | Professional Git workflow notes |
| `week3/` | API auth, LLM client, FastAPI webhook |
| `week4/` | Tool use, memory, AI assistant CLI |
| `phase2_week1/` | n8n workflows and automation |
| `phase2_week2/` | RAG pipeline and vector database |
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
| `week3/week3_day2_auth.py` | Authentication types and LLM API integration (GapGPT) |
| `week3/week3_day3_retry.py` | Rate limit handling with retry and exponential backoff |
| `week3/week3_day4_webhook.py` | FastAPI webhook endpoint with AI response handler |
| `week3/week3_project.py` | Multi-provider LLM client with fallback and retry logic |

### Week 4 — LLM Tool Use & First AI Agent
| File | Description |
|------|-------------|
| `week4/week4_day1_tools.py` | LLM tool use with ReAct agent loop |
| `week4/week4_day2_multi_tool.py` | Multi-tool agent with system prompt and knowledge base |
| `week4/week4_day3_memory.py` | Agent with conversation memory |
| `week4/week4_day4_prompts.py` | Prompt engineering examples with structured output |
| `week4/week4_day5_error_handling.py` | Safe agent with error handling and max iterations guard |
| `week4/week4_project.py` | Complete AI assistant CLI with tools, memory, and fallback |

### Phase 2 Week 1 — n8n Fundamentals
| Workflow | Description |
|----------|-------------|
| `n8n-workflows/phase2_week1_first_workflow.json` | First n8n workflow — HTTP Request basics |
| `n8n-workflows/phase2_week1_error_handling.json` | Error handling with On Error and IF node |
| `n8n-workflows/phase2_week1_code_node.json` | Data processing with JavaScript Code node |
| `n8n-workflows/phase2_week1_telegram.json` | Telegram integration — send API posts |
| `n8n-workflows/phase2_week1_telegram_bot.json` | Telegram bot — receive and reply to messages |
| `n8n-workflows/phase2_week1_news_digest.json` | Daily news digest with GapGPT summarization |

### Phase 2 Week 2 — RAG & Vector Database
| File | Description |
|------|-------------|
| `phase2_week2/day1_embeddings.py` | Text embeddings and semantic search with cosine similarity |
| `phase2_week2/day2_chroma.py` | Persistent vector storage with ChromaDB |
| `phase2_week2/day3_rag_pipeline.py` | Complete RAG pipeline: embed → search → LLM answer |
| `phase2_week2/day4_chunking.py` | Document chunking strategies (paragraph vs size-based) |
| `phase2_week2/day5_advanced_rag.py` | Advanced RAG with metadata filtering and auto-category detection |
| `phase2_week2/day6_project_qa_bot.py` | Complete customer support Q&A bot with RAG and memory |

## 🚀 Setup

```bash
git clone https://github.com/sina-naderi/ai-agent-journey.git
cd ai-agent-journey
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```
GAP_API_KEY=your_gapgpt_key_here
MY_NAME=YourName
```

## 🤖 Run the AI Assistant

```bash
python week4/week4_project.py
```

## 🧠 Run the Q&A Bot (RAG)

```bash
python phase2_week2/day6_project_qa_bot.py
```

## 📈 Progress

- [x] Week 1 — Python Foundations (async, APIs, error handling)
- [x] Week 2 — Professional Git (branching, PR, conflict resolution)
- [x] Week 3 — API Auth, LLM Integration, FastAPI Webhook
- [x] Week 4 — LLM Tool Use, Memory, First AI Agent
- [x] Phase 2 Week 1 — n8n Fundamentals (workflows, Telegram, GapGPT)
- [x] Phase 2 Week 2 — RAG & Vector Database (ChromaDB, embeddings, Q&A bot)
- [ ] Phase 2 Week 3 — LangChain & LangGraph
- [ ] Phase 2 Week 4 — PostgreSQL, FastAPI & Capstone Agent
- [ ] Phase 3 — Portfolio Projects & Deployment
- [ ] Phase 4 — Career Stack & First Applications
- [ ] Phase 5 — Interview Prep & System Design
