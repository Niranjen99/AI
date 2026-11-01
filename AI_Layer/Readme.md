# LiveWell AI Layer

The AI backend for **LiveWell**, an agentic mobile health-coaching app for older adults. This service powers the in-app AI coach: conversational chat, goal and reminder creation, frailty-based routines, push-notification content, motivational nudges, voice/text intent routing, and personalized resource recommendations.

It is a FastAPI service backed by a Groq-hosted LLM, with layered memory (Redis for short-term chat and profiles, ChromaDB for long-term summaries) and live weather/air-quality awareness.

---

## Table of Contents

1. [My Role in the Team](#1-my-role-in-the-team)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Setup](#6-setup)
7. [Running the Server](#7-running-the-server)
8. [API Reference](#8-api-reference)
9. [Memory System](#9-memory-system)
10. [Weather and Safety Engine](#10-weather-and-safety-engine)
11. [Proactive Profile Questions](#11-proactive-profile-questions)
12. [Recent Updates](#12-recent-updates)
13. [Roadmap](#13-roadmap)

---

## 1. My Role in the Team

LiveWell was built by a team of three:

| Member | Area | Responsibility |
|--------|------|----------------|
| **Anuja** | Mobile app (frontend) | Expo / React Native client: screens (home, dashboard, profile, leaderboard), chat UI, goal tracking, notifications |
| **Sonish** | App backend and data | Node.js backend with MongoDB: users, action templates, progress logs, resources, core REST API |
| **Me** | AI / agentic layer | This repository: everything that makes the app *agentic* |

As the **AI / agentic layer engineer**, I designed and built:

- **The AI health coach agent** (`/chat`): a single LLM call that decides whether the user is chatting, creating a goal, or setting a reminder, and returns structured JSON the mobile app can act on directly.
- **The memory architecture**: short-term chat history and a structured user profile in Redis, plus an evolving long-term summary stored as embeddings in ChromaDB, combined into one context window per request.
- **Automatic profile extraction**: relevance filtering of user messages, LLM-based profile updates with contradiction handling (likes / dislikes / goals / health notes / location).
- **Proactive profile building**: an engine that finds gaps across the five AVOID frailty-prevention categories and weaves one natural follow-up question into the coach's replies, spaced out so it never feels like an interrogation.
- **Context-aware safety**: weather, air-quality, wind and time-of-day analysis (with Redis caching) so outdoor-activity advice is safe for older adults and scaled to frailty level.
- **Frailty routine generation**: converting frailty questionnaire answers into 4-6 structured, categorised daily actions.
- **Dynamic notifications and nudges**: personalized push-notification text and progressive goal suggestions based on streaks, completed and missed actions, weather and memory.
- **Intent detection**: classifying user input into "navigate to screen X" or "chat" so the app can route voice/text commands.
- **Personalized resources**: LLM ranking of the resource library against each user's profile and history.
- **Prompt engineering and robustness**: strict JSON output contracts, markdown stripping, recovery from Groq `tool_use_failed` errors, and safe fallbacks on every endpoint.
- **Deployment**: Dockerfile, docker-compose, and ngrok setup for integration with the mobile team.

---

## 2. Features

| Feature | Endpoint | Description |
|---------|----------|-------------|
| AI coach chat | `POST /chat` | Conversational coach that also creates goals and reminders as structured actions |
| Frailty routine | `POST /frailty_routine` | Turns frailty questionnaire answers into 4-6 structured daily actions |
| Next-action notification | `POST /notifications/nextaction` | Personalized notification text plus an optional new or progressed goal |
| Nudges | `POST /nudges` | 1-2 short motivational nudges from memory and recent MongoDB activity logs |
| Intent detection | `POST /intent` | Classifies text as screen navigation or chat |
| Personalized resources | `GET /resources/personalized` | Ranks articles/resources by relevance to the user |

All prompts follow the **AVOID** frailty-prevention framework:
**A**ctivity, **V**accination, **O**ptimising medication, **I**nteraction & socialisation, **D**iet & nutrition.

---

## 3. Architecture

```
          Mobile app (React Native / Expo)
                        │  HTTPS (JSON)
                        ▼
┌────────────────────── FastAPI (main.py) ──────────────────────┐
│ /chat  /frailty_routine  /notifications/nextaction            │
│ /nudges  /intent  /resources/personalized                     │
└───────┬───────────────┬──────────────────┬───────────────┬────┘
        │               │                  │               │
        ▼               ▼                  ▼               ▼
  memory.py     proactive_questions.py  weather.py     Groq LLM
  ├─ Redis: chat history,   ├─ AVOID gap     ├─ WeatherAPI   (openai/gpt-oss-120b
  │  profiles, counters     │  analysis      │  + AQI         via langchain_groq)
  └─ ChromaDB: long-term    └─ question      └─ Redis cache
     summaries (MiniLM)        generation       (30 min)
                        │
                        ▼
                MongoDB (user_logs, resources)
```

Request flow for `/chat`:

1. Increment the message counter used to space out profile questions.
2. Load memory: last 4 messages + profile + top long-term summaries.
3. If the message is weather or activity related, fetch weather for the user's saved location and run the safety analysis.
4. Optionally generate one proactive profile question for a weak AVOID category.
5. Build the coach prompt and call the LLM, which must return JSON.
6. Recover from tool-call errors, strip markdown, parse JSON.
7. Save both sides of the conversation, which may trigger a profile update and, every 5 messages, a summary update.

---

## 4. Tech Stack

- **API:** FastAPI, Uvicorn, Pydantic
- **LLM:** Groq (`openai/gpt-oss-120b`) via `langchain_groq`
- **Short-term memory and cache:** Redis
- **Long-term memory:** ChromaDB with HuggingFace `all-MiniLM-L6-v2` embeddings
- **App data:** MongoDB (`pymongo`)
- **Weather and AQI:** WeatherAPI.com
- **Deployment:** Docker, docker-compose, ngrok

---

## 5. Project Structure

```
AI_Layer/
├── main.py                   # FastAPI app, all endpoints and prompts
├── memory.py                 # Redis + ChromaDB memory, profile extraction, summaries
├── proactive_questions.py    # AVOID coverage analysis and question generation
├── weather.py                # Weather/AQI fetch, caching, safety analysis, timezone
├── chroma.py                 # Debug script to query the Chroma vector store
├── chroma_db/                # Local persisted Chroma database
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Request.txt               # Sample PowerShell requests for /chat and /notifications/nextaction
├── bonusrequest.txt          # Sample /notifications/nextaction request
└── samplefrailtyrequest.txt  # Sample /frailty_routine request
```

---

## 6. Setup

### 6.1 Prerequisites

- Python 3.11 (the Docker image uses `python:3.11-slim`)
- Redis running locally or remotely
- MongoDB with a `livewell` database (collections `user_logs` and `resources`)
- A Groq API key and a WeatherAPI.com API key

### 6.2 Virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 6.3 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs the CPU-only build of PyTorch (`torch==2.3.1+cpu`) to keep the image small for the embedding model.

### 6.4 Environment variables

Create a `.env` file in `AI_Layer/` (never commit it):

```env
GROQ_API_KEY=your_groq_key
WEATHER_API_KEY=your_weatherapi_key

MONGO_URI=mongodb://localhost/livewell
MONGO_DB=livewell

REDISHOST=localhost
REDISPORT=6379
REDISPASSWORD=
```

| Variable | Used in | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | `main.py` | none (required) |
| `WEATHER_API_KEY` | `weather.py` | none (required for weather features) |
| `MONGO_URI` | `main.py` | `mongodb://localhost/livewell` |
| `MONGO_DB` | `main.py` | `livewell` |
| `REDISHOST` | `memory.py`, `weather.py`, `proactive_questions.py` | `localhost` |
| `REDISPORT` | same | `6379` |
| `REDISPASSWORD` | same | none |

---

## 7. Running the Server

### 7.1 Local

```bash
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

### 7.2 Docker

```bash
docker compose up --build
```

This starts the API on port 8000 and Redis on port 6379. MongoDB is not included and must be reachable through `MONGO_URI`.
### 7.3 Public URL for the mobile app (ngrok)

```bash
ngrok http 8000
```

Give the generated HTTPS URL to the mobile team. CORS is open (`allow_origins=["*"]`) for development.

---

## 8. API Reference

### 8.1 Chat: `POST /chat`

```json
{ "uid": "user_123", "text": "Can I go for a walk today?" }
```

The response is always one of three shapes.

Chat:
```json
{ "type": "chat", "message": "It's a lovely 22 degree morning, perfect for a 15 minute walk. Do you usually walk alone or with friends?" }
```

Goal / action:
```json
{
  "type": "action",
  "payload": {
    "title": "Morning walk",
    "description": "Walk at a comfortable pace",
    "category": "Activity",
    "target_value": "15",
    "target_unit": "minutes",
    "frequency": "Daily",
    "difficulty_level": "Easy"
  },
  "message": "Great, I've set up a daily 15 minute walk for you."
}
```

Reminder:
```json
{
  "type": "reminder",
  "payload": { "title": "Take blood pressure tablet", "date_time": "2025-10-14T08:00:00" },
  "message": "I'll remind you tomorrow at 8am."
}
```

On an unexpected server error the response is `{ "error": "..." }`.

### 8.2 Frailty routine: `POST /frailty_routine`

```json
{
  "uid": "user_123",
  "frailtyScore": 4,
  "responses": [
    { "label": "Fatigue: How much of the time during the past 4 weeks did you feel tired?", "value": "All or most of the time" },
    { "label": "Resistance: Do you have any difficulty walking up 10 steps without resting?", "value": "Yes" }
  ]
}
```

Response:
```json
{
  "uid": "user_123",
  "frailtyScore": 4,
  "actions": [
    {
      "title": "Daily hydration",
      "description": "Drink water steadily through the day",
      "category": "Diet & nutrition",
      "target_value": "2",
      "target_unit": "L",
      "frequency": "Daily",
      "difficulty_level": "Easy"
    }
  ]
}
```

Full sample: `samplefrailtyrequest.txt`.

### 8.3 Next-action notification: `POST /notifications/nextaction`

```json
{
  "userId": "user_698",
  "lastCompletedAction": {
    "action_template_id": "68c986e06c9143419251cdb5",
    "title": "Short Indoor Walk",
    "description": "Walk at a comfortable pace for 5 minutes inside your home.",
    "category": "Activity",
    "target_value": 5,
    "target_unit": "minutes",
    "frequency": "daily",
    "difficulty_level": "Beginner",
    "completedAt": "2025-10-13T11:33:01.055Z"
  },
  "missedActions": [
    {
      "action_template_id": "68c986e06c9143419251cdc0",
      "title": "Hydration Reminder",
      "description": "Drink at least 1 glass of water every hour",
      "frequency": "daily",
      "difficulty_level": "Beginner",
      "dueDate": "2025-10-13T09:00:00.000Z"
    }
  ],
  "context": {
    "timeSinceLastActionMinutes": 60,
    "streakDays": 3,
    "location": { "lat": -34.9212, "lng": 138.5995 }
  }
}
```

Response (`type` is one of `encourage`, `update`, `create`, `bonus`; `goal` is present for every type except `encourage`):

```json
{
  "text": "Three days in a row, well done! Ready to stretch your walk to 8 minutes today?",
  "type": "update",
  "goal": {
    "action_template_id": "68c986e06c9143419251cdb5",
    "title": "Short Indoor Walk",
    "description": "Walk at a comfortable pace for 8 minutes inside your home.",
    "category": "Activity",
    "target_value": 8,
    "target_unit": "minutes",
    "frequency": "daily",
    "difficulty_level": "Easy",
    "next_due_date": "2025-10-14T10:00:00Z"
  }
}
```

If anything fails, the endpoint returns a safe `encourage` message instead of an error.

### 8.4 Nudges: `POST /nudges`

```json
{ "uid": "user_123" }
```

Uses memory plus the last 2 days of `action_progress` entries from MongoDB `user_logs`.

```json
{ "nudges": "You've walked three days in a row. Keep the streak going with another short walk today." }
```

### 8.5 Intent detection: `POST /intent`

```json
{ "text": "Show me my dashboard" }
```

```json
{ "intent": "Navigate", "screen": "dashboard" }
```
or
```json
{ "intent": "Chat" }
```

Supported screens: `home`, `dashboard`, `profile`, `Leaderboard`.

### 8.6 Personalized resources: `GET /resources/personalized?uid=user_123&top_k=6`

Returns up to `top_k` documents from the MongoDB `resources` collection, ranked by the LLM against the user's memory and profile, with newer resources preferred when relevance is similar.

---

## 9. Memory System

Implemented in `memory.py`.

| Layer | Store | Key / tag | Contents |
|-------|-------|-----------|----------|
| Short-term | Redis list | `chat:{uid}` | Last 5 messages (`role: content`) |
| Profile | Redis string (JSON) | `profile:{uid}` | `likes`, `dislikes`, `goals`, `health_notes`, `location` |
| Long-term | ChromaDB | metadata `tag={uid}_summary` | One evolving summary per user |
| Counter | Redis | `counter:{uid}` | Triggers a summary update every 5 saved messages |

- `save_message()` stores the message. For user messages it runs `is_relevant_message()` (filler filter, keywords, word count, then LLM fallback) and, if relevant, `update_profile()`.
- `update_profile()` extracts location and asks the LLM to merge new facts while removing contradictions, for example moving an item from likes to dislikes.
- `update_summary()` merges the previous Chroma summary with recent messages, replaces the old document, and persists it.
- `get_memory()` returns short-term messages, the profile block and the semantically closest long-term summaries as a single context list.

---

## 10. Weather and Safety Engine

Implemented in `weather.py`.

- Weather is fetched **only** when the message is about weather or planning an activity (`should_fetch_weather`), to save latency and API calls.
- The location comes from the user's profile and falls back to `Adelaide, AU`.
- Results are cached in Redis for 30 minutes (`weather:{location}`).
- Time of day uses the `Australia/Adelaide` timezone.

Safety rules applied by `analyze_weather_safety()`:

| Factor | Good | Caution | Unsafe |
|--------|------|---------|--------|
| Temperature | 15-28 °C | 10-35 °C | below 5 or above 35 °C |
| Air quality (US EPA index) | 1-3 | | 4-6 |
| Wind | under 7 m/s | 7-10 m/s | 10 m/s or more |
| Conditions | clear | | rain, storm, snow, thunder |
| Time | daytime | | night, before 7am, after 8pm |

The coach then scales its suggestion by frailty: fit users get 20-40 minutes, moderate 10-20 minutes, frail 5-10 minutes with supervision, and indoor alternatives (tai chi, chair exercises, stretching) when it is unsafe outside.

---

## 11. Proactive Profile Questions

Implemented in `proactive_questions.py`.

1. Keyword coverage is scored for each AVOID category across the profile and recent memory (`very_low`, `low`, `moderate`, `high`).
2. The weakest category is picked, rotating away from the one asked last time.
3. The LLM writes one short, warm, optional question that does not repeat earlier ones (stored in `user:{uid}:asked_questions`).
4. At least 3 messages must pass between questions (`user:{uid}:message_count_since_question`).
5. `/chat` asks the coach to weave the question naturally into the end of its reply.

---

## 12. Recent Updates

- **Weather provider moved to WeatherAPI.com** with built-in US EPA air quality, Redis caching, and better error logging.
- **Adelaide-local time of day** using `pytz`, so time-based outdoor safety is correct on UTC servers.
- **Structured weather safety analysis** passed to the LLM instead of raw numbers, so responses read naturally ("a lovely 22 degree day").
- **Location-aware profiles**: users' cities are extracted from chat and used for weather.
- **New `/notifications/nextaction` endpoint** for push-notification text with progressive goal suggestions (`encourage`, `update`, `create`, `bonus`).
- **Proactive AVOID questions** in chat for gradual profile building, with spacing and de-duplication.
- **Configurable Redis** through `REDISHOST`, `REDISPORT` and `REDISPASSWORD` for hosted deployments.
- **MongoDB settings from environment** (`MONGO_URI`, `MONGO_DB`).
- **Robust LLM output handling**: recovery from Groq `tool_use_failed` errors, markdown code-fence stripping, and fallbacks on every endpoint.
- **Plain-text output rule** (no emojis or fancy unicode) for consistent rendering in the mobile app.
- **Model switched** to `openai/gpt-oss-120b` on Groq.
- **CORS enabled** for the Expo mobile and web clients.
- **CPU-only PyTorch** in requirements for smaller Docker images.

---

## 13. Roadmap

- Read the frailty score from the database (test values are used during early development while user data is collected)
- Per-user timezone from location
- Embedding-based pre-filtering for resources before LLM ranking
- Auth (e.g. Firebase ID token verification) and restricted CORS
- Unit tests for the safety engine and JSON parsing, plus endpoint integration tests
- Proper logging instead of `print` statements
- Split `main.py` into routers (chat, notifications, resources)
