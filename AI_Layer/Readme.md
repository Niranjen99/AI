# LiveWell AI Layer

Backend for LiveWell project. Handles user chat, frailty routines, nudges, intent detection, profile updates, and personalized resource suggestions using an LLM.

---

## 1. Setup

### 1.1 Create Virtual Environment

```bash
python -m venv venv
# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Configure Environment

Edit `main.py` if needed:
* **Redis host/port**: update where `redis_client` is initialized
* **LLM API key**: update `ChatGroq(groq_api_key="YOUR_KEY")`
* **MongoDB URI**: update in `MongoClient("mongodb://localhost/livewell")`

## 2. Run the Server

### 2.1 Local

```bash
uvicorn main:app --reload
```

Access at `http://127.0.0.1:8000`

### 2.2 Optional: Ngrok (for public URL)

```bash
ngrok http 8000
```

Use the generated URL to access endpoints remotely.

## 3. Endpoints & Example Usage

### 3.1 Chat

**POST** `/chat`

```json
{
  "uid": "user_123",
  "text": "How is my health profile?"
}
```

Returns AI conversational response or goal/reminder JSON.

### 3.2 Frailty Routine

**POST** `/frailty_routine`

```json
{
  "uid": "user_123",
  "frailtyScore": 5,
  "responses": [
    {"label": "Walking", "value": "15 minutes"},
    {"label": "Hydration", "value": "2 liters"}
  ]
}
```

Returns structured suggested actions.

### 3.3 Nudges

**POST** `/nudges`

```json
{
  "uid": "user_123"
}
```

Returns 1-2 motivational nudges.

### 3.4 Intent Detection

**POST** `/intent`

```json
{
  "text": "Show me my dashboard"
}
```

Returns intent JSON:

```json
{"intent": "Navigate", "screen": "dashboard"}
```

### 3.5 Personalized Resources

**GET** `/resources/personalized?uid=user_123&top_k=6`

Returns top 5–6 resources based on user profile, chat, and semantic summary.

## 4. Notes

* Redis and MongoDB must be running.
* `get_memory()` combines short-term messages, profile, and long-term summary.
* LLM usage may hit token limits if resources list is large.