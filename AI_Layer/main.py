from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional
from langchain_groq import ChatGroq
from memory import save_message, get_memory
from proactive_questions import generate_next_question, increment_message_count
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

# Allow requests from all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Or ["http://localhost:19006"] for Expo Web
    allow_methods=["*"],        # Allow POST, OPTIONS, GET, etc.
    allow_headers=["*"],        # Allow custom headers
)

import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b"
)

class ChatRequest(BaseModel):
    uid: str
    text: str

class FrailtyResponseItem(BaseModel):
    label: str
    value: str

class FrailtyRequest(BaseModel):
    uid: str
    frailtyScore: int
    responses: List[FrailtyResponseItem]


from typing import Union

class ActionItem(BaseModel):
    title: str
    description: str
    category: str
    target_value: Optional[Union[str, int]] = None   # Accept string OR int
    target_unit: Optional[str] = None
    frequency: Optional[str] = None
    difficulty_level: Optional[str] = None



class FrailtyResponse(BaseModel):
    uid: str
    frailtyScore: int
    actions: List[ActionItem]


class NudgeRequest(BaseModel):
    uid: str

class IntentRequest(BaseModel):
    text: str

# Hardcoded dummy frailty scores for now
dummy_frailty_scores = {
    "user_123": 5,   # moderately frail
    "user_456": 2    # quite fit
}


import json
from weather import (
    get_weather, 
    parse_aqi_from_weather,
    analyze_weather_safety,  # New function
    should_fetch_weather,
    get_user_location,
    get_time_of_day
)

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Increment message counter for question spacing
        increment_message_count(req.uid)
        
        # Get conversation context
        past_messages = get_memory(req.uid, req.text, short_term_limit=4, long_term_k=3)
        context = "\n".join(past_messages)

        frailty_score = dummy_frailty_scores.get(req.uid, 3)
        
        # ===== WEATHER INTEGRATION =====
        weather_context = ""
        
        if should_fetch_weather(req.text):
            # Get user's location from profile
            user_location = get_user_location(req.uid)
            
            # Fetch weather data
            weather_data = get_weather(user_location)
            
            if weather_data:
                # Get structured weather analysis
                aqi_data = parse_aqi_from_weather(weather_data)
                weather_analysis = analyze_weather_safety(weather_data, aqi_data)
                period, hour = get_time_of_day()
                
                # Build weather context for AI
                weather_context = f"""
[Current Weather Analysis for {weather_data['location']}]:
Temperature: {weather_analysis['temperature']['value']:.1f}°C
- Status: {weather_analysis['temperature']['status']}
- Concerns: {', '.join(weather_analysis['temperature']['concerns']) if weather_analysis['temperature']['concerns'] else 'None'}

Air Quality: {weather_analysis['air_quality']['label']}
- AQI Level: {weather_analysis['air_quality']['aqi']}
- Status: {weather_analysis['air_quality']['status']}

Wind: {weather_analysis['wind']['speed']:.1f} m/s
- Status: {weather_analysis['wind']['status']}

Conditions: {weather_analysis['conditions']['description']}
- Status: {weather_analysis['conditions']['status']}

Time: {period} ({hour}:00)
- Outdoor safe for elderly: {weather_analysis['time_safety'][0]}
{f"- Time concern: {weather_analysis['time_safety'][1]}" if not weather_analysis['time_safety'][0] else ""}

Overall Safety for Outdoor Activity: {"✓ SAFE" if weather_analysis['overall_safe'] else "✗ NOT RECOMMENDED"}
"""
        # ==============================
        
        # === DYNAMIC QUESTION GENERATION ===
        profile_question_data = generate_next_question(req.uid)
        question_prompt = ""
        
        if profile_question_data:
            question_prompt = f"""
**IMPORTANT - Profile Building:**
At the end of your response, naturally and subtly include this question:
"{profile_question_data['question']}"

Integration guidelines:
- Make it feel like a natural continuation of the conversation
- Don't force it if it doesn't fit the context
- Keep it casual and optional
- The user should feel comfortable answering or ignoring it
"""
        # ==================================
        
        frailty_context = f"""
You are LiveWell, a supportive health coach for older adults.
- Always encourage safe, positive health habits (Activity, Vaccination, Medication, Interaction, Diet & nutrition).
- Keep answers short and conversational (3–4 sentences max).
- Frailty score: {frailty_score} (1-3=fit, 4-6=moderate, 7-10=frail)
- Never give unsafe medical advice.
- Detect if the user is trying to create a goal, set a reminder, or just chat.
- IMPORTANT: Use plain text only - NO emojis, NO special unicode characters, NO fancy quotes or dashes

{weather_context}

{question_prompt}

**Weather-Based Recommendations (only when user asks for it or something related to it where weather is relevant):**
- Use the structured weather analysis above to give natural, contextual advice
- If overall safety is NOT RECOMMENDED: Strongly suggest safe indoor alternatives (yoga, tai chi, chair exercises, stretching)
- If temperature status is "unsafe": Mention the specific concern (extremely hot/cold) and recommend staying indoors
- If air quality status is "unsafe": Explain air quality issue and suggest indoor activities
- If time is not safe (nighttime/very early): Mention it's not a good time and suggest indoor options or waiting
- If wind status is "unsafe": Mention windy conditions make it unsafe
- If conditions status is "unsafe": Mention rain/storm/snow and suggest indoor alternatives

**For SAFE conditions:**
- Frailty 1-3 (fit): Suggest 20-40 minutes of activity (walking, jogging, gardening, cycling)
- Frailty 4-6 (moderate): Suggest 10-20 minutes of gentle activity (short walks, light gardening)
- Frailty 7-10 (frail): Suggest 5-10 minutes of very gentle activity WITH supervision (sitting outside, very short walks)

**Tone:**
- Be warm, natural, and encouraging
- Weave weather details into your response naturally.e.g., "It's a beautiful 24°C day" instead of "Temperature: 24°C Status: good".(only when user asks for it or something related to it where weather is relevant)
- Prioritize safety while staying positive
- Match the user's energy and question style
- Dont mention the frailty score, as it may discourage the user.

**Response Format - Return ONLY valid JSON (no markdown, no code blocks):**

For goal/action, respond with:
{{"type": "action", "payload": {{"title": "Goal title", "description": "Description of goal", "category": "Activity/Diet/Medication/etc", "target_value": "numeric value", "target_unit": "minutes/glasses/etc", "frequency": "Daily/Weekly/etc", "difficulty_level": "Easy/Medium/Hard"}}, "message": "Dynamic message to user confirming action"}}

For reminder, respond with:
{{"type": "reminder", "payload": {{"title": "Reminder title", "date_time": "ISO 8601 datetime"}}, "message": "Dynamic message to user confirming reminder"}}

For normal chat, respond with:
{{"type": "chat", "message": "Your natural, conversational reply (with optional question at the end if provided)"}}

IMPORTANT: Return ONLY the JSON object, nothing else. No markdown formatting, no explanations.
"""

        prompt = f"{frailty_context}\n\nConversation history:\n{context}\n\nUser: {req.text}\nAI:"

        # Ask LLM to generate response with error handling for tool calling
        try:
            llm_response = llm.invoke(prompt).content.strip()
        except Exception as e:
            error_str = str(e)
            # If tool calling error, extract the failed generation
            if "tool_use_failed" in error_str and "arguments" in error_str:
                try:
                    # Extract the arguments portion
                    # Pattern: "arguments": {...}
                    start_idx = error_str.find('"arguments":')
                    if start_idx == -1:
                        start_idx = error_str.find("'arguments':")
                    
                    if start_idx != -1:
                        # Find the opening brace
                        brace_start = error_str.find('{', start_idx)
                        if brace_start != -1:
                            # Count braces to find matching closing brace
                            brace_count = 0
                            brace_end = brace_start
                            for i in range(brace_start, len(error_str)):
                                if error_str[i] == '{':
                                    brace_count += 1
                                elif error_str[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        brace_end = i
                                        break
                            
                            # Extract the JSON
                            arguments_json = error_str[brace_start:brace_end+1]
                            # Clean up escaped characters
                            arguments_json = arguments_json.replace("\\'", "'").replace('\\n', ' ')
                            llm_response = arguments_json
                            print(f"✓ Extracted from tool error: {llm_response[:100]}...")
                        else:
                            raise ValueError("Could not find opening brace")
                    else:
                        raise ValueError("Could not find arguments field")
                        
                except Exception as parse_error:
                    print(f"✗ Failed to parse tool error: {parse_error}")
                    print(f"Error string: {error_str[:500]}...")
                    llm_response = json.dumps({
                        "type": "chat",
                        "message": "I'm here to help! How can I assist you today?"
                    })
            else:
                raise e
        
        # Remove markdown code blocks if present
        if llm_response.startswith("```"):
            llm_response = llm_response.split("```")[1]
            if llm_response.startswith("json"):
                llm_response = llm_response[4:]
            llm_response = llm_response.strip()

        # Parse and save to memory
        try:
            response_json = json.loads(llm_response)
        except json.JSONDecodeError:
            response_json = {
                "type": "chat",
                "message": llm_response
            }
        
        save_message(req.uid, "user", req.text)
        save_message(req.uid, "ai", response_json.get("message", llm_response))

        return response_json

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
#frailty endpoint to return structured actions
@app.post("/frailty_routine", response_model=FrailtyResponse)
async def frailty_routine(req: FrailtyRequest):
    try:
        # Save frailty score and responses to memory
        save_message(req.uid, "frailty", f"Frailty score: {req.frailtyScore}")
        for item in req.responses:
            save_message(req.uid, "frailty", f"{item.label}: {item.value}")

        # Fetch memory context
        context = "\n".join(get_memory(req.uid, "frailty routine"))

        # Prompt the LLM to return JSON-like structured actions
        prompt = f"""
You are LiveWell Coach. Based on the following frailty responses and user history, suggest 4-6 safe daily actions. Return ONLY valid JSON with this structure (no extra text):
{{
    "actions": [
        {{
            "title": "<short title>",
            "description": "<detailed explanation>",
            "category": "<one of Activity, Vaccination, Optimising medication, Interaction & socialisation, Diet & nutrition>",
            "target_value": "<number if applicable>",
            "target_unit": "<unit if applicable>",
            "frequency": "<how often>",
            "difficulty_level": "<Beginner|Easy|Moderate|Challenging>"
        }}
    ]
}}

Instructions:
- Only use the five AVOID categories above.
- Ensure category is mapped correctly, if its walking its not socialisation its activity
- Hydration actions → map to "Diet & nutrition"-> also use litres(L) instead of millilitres(mL) for the unit.
- Ensure each action has a clear, simple description.
- Provide numeric targets as strings, e.g., "15", "2".
- Always suggest target unit and target value if its some like exercise or hydration, or something you can suggest without professional help.
- Frequency and difficulty must always be filled.
- Provide fixed numbers for frequency and target_valus
Frailty Score: {req.frailtyScore}
Frailty Responses:
{context}
"""
        raw_response = llm.invoke(prompt).content

        import json
        try:
            actions_data = json.loads(raw_response)
        except Exception:
            actions_data = {"actions": []}

        return {
            "uid": req.uid,
            "frailtyScore": req.frailtyScore,
            "actions": actions_data.get("actions", [])
        }
    except Exception as e:
        return {"error": str(e)}


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
import json

# Add these models to your existing code

class Location(BaseModel):
    lat: float
    lng: float

class LastCompletedAction(BaseModel):
    action_template_id: str
    title: str
    description: str
    category: str
    target_value: int
    target_unit: str
    frequency: str
    difficulty_level: str
    completedAt: str

class MissedAction(BaseModel):
    action_template_id: str
    title: str
    description: str
    frequency: str
    difficulty_level: str
    dueDate: str

class ContextInfo(BaseModel):
    timeSinceLastActionMinutes: int
    streakDays: int
    location: Optional[Location] = None

class DynamicNotificationRequest(BaseModel):
    userId: str
    lastCompletedAction: Optional[LastCompletedAction] = None
    missedActions: List[MissedAction] = []
    context: ContextInfo

class GoalSuggestion(BaseModel):
    action_template_id: str
    title: str
    description: str
    category: str
    target_value: int
    target_unit: str
    frequency: str
    difficulty_level: Literal["Beginner", "Easy", "Intermediate", "Hard", "Advanced"]
    next_due_date: str

class DynamicNotificationResponse(BaseModel):
    text: str
    type: Literal["update", "create", "encourage", "bonus"]
    goal: Optional[GoalSuggestion] = None

# Add this endpoint to your existing FastAPI app

@app.post("/notifications/nextaction", response_model=DynamicNotificationResponse)
async def dynamic_notification(req: DynamicNotificationRequest):
    try:
        # Get user memory context
        memory_context = "\n".join(get_memory(req.userId, "notification context", short_term_limit=5, long_term_k=3))
        
        # Get frailty score
        frailty_score = dummy_frailty_scores.get(req.userId, 3)
        
        # Get weather data for user's location
        weather_context = ""
        user_location = get_user_location(req.userId)
        weather_data = get_weather(user_location)
        
        if weather_data:
            aqi_data = parse_aqi_from_weather(weather_data)
            weather_analysis = analyze_weather_safety(weather_data, aqi_data)
            weather_context = f"""
Current Weather ({weather_data['location']}):
- Temperature: {weather_analysis['temperature']['value']:.1f}°C ({weather_analysis['temperature']['status']})
- Air Quality: {weather_analysis['air_quality']['label']} (AQI: {weather_analysis['air_quality']['aqi']})
- Conditions: {weather_analysis['conditions']['description']} ({weather_analysis['conditions']['status']})
- Overall Safety: {"SAFE" if weather_analysis['overall_safe'] else "NOT RECOMMENDED"}
"""
        
        # Build context string
        last_action_str = ""
        if req.lastCompletedAction:
            last_action_str = f"""
Last Completed Action:
- Title: {req.lastCompletedAction.title}
- Category: {req.lastCompletedAction.category}
- Target: {req.lastCompletedAction.target_value} {req.lastCompletedAction.target_unit}
- Difficulty: {req.lastCompletedAction.difficulty_level}
- Completed: {req.lastCompletedAction.completedAt}
"""
        
        missed_actions_str = ""
        if req.missedActions:
            missed_actions_str = "Missed Actions:\n"
            for action in req.missedActions:
                missed_actions_str += f"- {action.title} (due: {action.dueDate})\n"
        
        context_str = f"""
Time since last action: {req.context.timeSinceLastActionMinutes} minutes
Current streak: {req.context.streakDays} days)
"""
        #Location: ({req.context.location.lat}, {req.context.location.lng}   
        # Build LLM prompt
        prompt = f"""
You are LiveWell Coach, a supportive health assistant for older adults. Generate a personalized notification based on the user's recent activity, context, and weather conditions.

User Profile & Memory:
{memory_context}

Frailty Score: {frailty_score} (1-3=fit, 4-6=moderate, 7-10=frail)

{last_action_str}

{missed_actions_str}

{context_str}

{weather_context}

**Guidelines:**
1. Be warm, encouraging, and personalized
2. Consider the user's frailty level when suggesting activities
3. Factor in weather safety (only suggest outdoor activities if weather is SAFE)
4. Acknowledge completed actions and streaks positively
5. Gently encourage missed actions
6. Suggest progressive goals (slightly harder than last completed action)
7. Time-aware: Consider how long since last action
8. IMPORTANT: Use plain text only - NO emojis, NO special unicode characters, NO fancy quotes or dashes

**Notification Types:**
- "encourage": Just motivation, no goal (use for very recent completions or when conditions aren't right)
- "update": Suggest increasing difficulty/duration of existing action
- "create": Suggest a new complementary action
- "bonus": Celebrate streaks or milestones with optional challenge

**Response Format (ONLY valid JSON):**
{{
  "text": "<warm, personalized notification message>",
  "type": "encourage|update|create|bonus",
  "goal": {{  // Only include if type is "update", "create", or "bonus"
    "action_template_id": "<existing_id or generate new one like 'new_walk_45min'>",
    "title": "<action title>",
    "description": "<detailed description>",
    "category": "<Activity|Diet & nutrition|Optimising medication|Interaction & socialisation|Vaccination>",
    "target_value": <number>,
    "target_unit": "<minutes|glasses|steps|etc>",
    "frequency": "daily|weekly|twice_weekly",
    "difficulty_level": "Beginner|Easy|Intermediate|Hard|Advanced",
    "next_due_date": "<ISO 8601 datetime for next suggested completion>"
  }}
}}

Generate notification now:
"""
        
        # Get LLM response
        llm_response = llm.invoke(prompt).content.strip()
        
        # Parse JSON response
        try:
            response_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Fallback response
            response_data = {
                "text": "Great work! Keep up the amazing progress. 🌟",
                "type": "encourage"
            }
        
        # Save notification to memory
        save_message(req.userId, "notification", f"Sent: {response_data.get('text', '')}")
        
        return response_data
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        # Fallback response on error
        return {
            "text": "You're doing great! Keep up the good work.",
            "type": "encourage"
        }

# Nudges endpoint 


from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

@app.post("/nudges")
async def nudges(req: NudgeRequest):
    try:
        # --- 1. Short + long-term memory ---
        memory_context = "\n".join(get_memory(req.uid, "personalized nudges"))

        # --- 2. Recent MongoDB logs ---
        # client = MongoClient("mongodb://localhost/livewell")
        # db = client["livewell"]
        # logs_col = db["user_logs"]

        load_dotenv()

        MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost/livewell")
        MONGO_DB = os.getenv("MONGO_DB", "livewell")

        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        logs_col = db["user_logs"]

        two_days_ago = datetime.utcnow() - timedelta(days=2)
        recent_logs = list(logs_col.find({
            "user_id": req.uid,
            "type": "action_progress",
            "createdAt": {"$gte": two_days_ago}
        }).sort("createdAt", -1).limit(5))

        logs_context = ""
        for log in recent_logs:
            metadata = log.get("metadata", {})
            title = metadata.get("title", "Action")
            progress = metadata.get("progress", 0)
            total = metadata.get("total", 0)
            logs_context += f"- {title}: {progress}/{total} completed.\n"

        if not logs_context:
            logs_context = "- No recent activity logged.\n"

        # --- 3. Combine context ---
        full_context = f"{memory_context}\nRecent activity:\n{logs_context}"

        # --- 4. Build LLM prompt ---
        prompt = f"""
You are a friendly, motivating Health Coach. Based on the user's profile, long-term memory, and recent actions the user's been following, generate 1-2 short motivational nudges or notifications. Keep it positive, actionable, and safe.
for Example if the user has been a on streak of walking daily motivate him to do the same today.
Context:
{full_context}
Output only 1-2 short sentences..
"""

        response = llm.invoke(prompt).content.strip()

        return {"nudges": response}

    except Exception as e:
        return {"error": str(e)}


    
@app.post("/intent")
async def detect_intent(req: IntentRequest):
    try:
        prompt = f"""
You are an intent classifier. Determine if the user wants to navigate to a screen or chat.
Example - "how is my profile looking like" (should go to profile page)
Screens: home, dashboard, profile, Leaderboard (case sensitive)
Respond ONLY in JSON format exactly like:
{{ "intent": "Navigate", "screen": "dashboard" }} or {{ "intent": "Chat" }}
User text: "{req.text}"
"""
        # Call Groq LLM
        response = llm.invoke(prompt).content  # returns string directly

        # Parse JSON safely
        import json
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if LLM returns slightly invalid JSON
            # Basic heuristic: check for keywords
            text_lower = req.text.lower()
            if "dashboard" in text_lower:
                result = {"intent": "Navigate", "screen": "dashboard"}
            elif "profile" in text_lower:
                result = {"intent": "Navigate", "screen": "profile"}
            elif "home" in text_lower:
                result = {"intent": "Navigate", "screen": "home"}
            elif "Leaderboard" in text_lower:
                result = {"intent": "Navigate", "screen": "Leaderboard"}
            else:
                result = {"intent": "Chat"}

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"intent": "Chat"}


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost/livewell")
MONGO_DB = os.getenv("MONGO_DB", "livewell")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
resources_col = db["resources"]

class ResourceItem(BaseModel):
    _id: str
    resourceType: str
    category: str
    title: str
    summary: str
    content: str = None
    imageUrl: str = None
    url: str = None
    source: str = None
    createdAt: str = None
    updatedAt: str = None

@app.get("/resources/personalized", response_model=List[ResourceItem])
async def personalized_resources(uid: str, top_k: int = 6):
    try:
        # --- 1. User context ---
        person_context = "\n".join(get_memory(uid, ""))
        

        # --- 2. Fetch resources ---
        resources = list(resources_col.find())
        if not resources:
            return []

        # --- 3. Minimal resource info for LLM ---
        resource_list = [
            {
                "_id": str(r["_id"]),
                "title": r.get("title", ""),
                "summary": r.get("summary", ""),
                "createdAt": r.get("createdAt", "")
            } 
            for r in resources
        ]

        # --- 4. LLM ranking prompt ---
        prompt = f"""
You are a helpful health coach assistant.
Rank the following resources for a user based on relevance to their profile, health notes, goals, and chat history.
Prefer more recent resources if relevance is similar.

Chat history and profile: {person_context}

Resources:
{json.dumps(resource_list)}

Return ONLY a JSON array of resource _ids (strings) in order from most relevant to least relevant.
"""

        ranked_ids_response = llm.invoke(prompt).content.strip()
        try:
            ranked_ids = json.loads(ranked_ids_response)
        except Exception:
            ranked_ids = [r["_id"] for r in resource_list]  # fallback

        # --- 5. Pick top_k ---
        top_resources = []
        seen_ids = set()
        for rid in ranked_ids:
            for r in resources:
                if str(r["_id"]) == rid and rid not in seen_ids:
                    r["_id"] = str(r["_id"])
                    top_resources.append(ResourceItem(**r))
                    seen_ids.add(rid)
                    break
            if len(top_resources) >= top_k:
                break

        return top_resources

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

