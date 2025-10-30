import redis
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json


import os


# Use environment variables if available, fallback to localhost for local runs
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

# ---- Chroma (long-term memory for evolving summaries) ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# -------- Profile Management --------

def extract_location_from_message(text: str):
    """
    Extract location mentions from user messages.
    Uses LLM to detect if user mentions their city/location.
    
    Args:
        text: User message
    
    Returns:
        Location string or None
    """
    from main import llm
    
    text_lower = text.lower()
    
    # Quick keyword check
    location_indicators = ["live in", "i'm in", "from", "located in", "city"]
    if not any(indicator in text_lower for indicator in location_indicators):
        return None
    
    prompt = f"""
Extract the location (city and country) from this message. If no location is mentioned, return null.

Message: "{text}"

Return ONLY JSON: {{"location": "City, Country"}} or {{"location": null}}

Examples:
- "I live in Sydney" → {{"location": "Sydney, AU"}}
- "I'm from Melbourne Australia" → {{"location": "Melbourne, AU"}}
- "I like walking" → {{"location": null}}
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        data = json.loads(response)
        return data.get("location")
    except Exception:
        return None


# Modified the update_profile function to include location extraction
def update_profile(uid: str, text: str):
    from main import llm
    key = f"profile:{uid}"
    profile = json.loads(redis_client.get(key) or "{}")

    # Check for location mention
    location = extract_location_from_message(text)
    if location:
        profile["location"] = location

    # Build prompt for structured extraction + contradiction handling
    prompt = f"""
You are a health coach assistant. Update the user profile based on their new message.
User message: "{text}"

The profile has these sections:
- likes: things the user enjoys.
- dislikes: things the user dislikes or avoids.
- goals: health-related goals the user wants to achieve.
- health_notes: relevant health notes (e.g., allergies, conditions).
- location: user's city/country (if mentioned).

Rules:
- If the new message contradicts existing info, REMOVE the old info and replace with the new one.
- If the user retracts a preference (e.g., "I don't like X anymore"), remove it from likes.
- If the user now dislikes something they used to like, move it from likes → dislikes.
- Do not duplicate existing entries.
- Keep lists short phrases only.
- Only update location if explicitly mentioned.

Existing profile: {json.dumps(profile)}

Return ONLY JSON with the full updated profile.
"""

    try:
        response = llm.invoke(prompt).content.strip()
        updated_profile = json.loads(response)
        redis_client.set(key, json.dumps(updated_profile))
    except Exception as e:
        print(f"Profile update failed: {e}")

def get_profile(uid: str) -> str:
    """Return structured user profile for context injection."""
    key = f"profile:{uid}"
    profile = json.loads(redis_client.get(key) or "{}")
    if not profile:
        return ""
    return f"[User Profile]: {json.dumps(profile, indent=2)}"

# -------- Conversation Memory --------

import re
def is_relevant_message(text: str) -> bool:
    from main import llm
    """Decide if a message is relevant for updating profile/summary."""
    text = text.lower().strip()

    # Common irrelevant / filler messages
    trivial = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you",
               "good morning", "good night", "bye"}
    if text in trivial:
        return False

    # Quick keyword relevance check
    keywords = [
        "like", "love", "enjoy", "dislike", "hate", "allergic",
        "goal", "want to", "plan to", "remind", "reminder", "set", "schedule",
        "exercise", "walk", "diet", "drink", "eat", "food", "health"
    ]
    if any(kw in text for kw in keywords):
        return True

    # Word count fallback
    if len(re.findall(r"\w+", text)) >= 6:
        return True

    # --- LLM fallback for edge cases ---
    prompt = f"""
You are a classifier. Decide if the following user message is relevant for updating a health profile or long-term memory summary. 
Relevant means: it expresses a preference, goal, habit, health info, or something useful to remember. 
Irrelevant means: greetings, short acknowledgements, or small talk.

Message: "{text}"

Return ONLY JSON: {{"relevant": true}} or {{"relevant": false}}
"""
    try:
        result = llm.invoke(prompt).content.strip()
        data = json.loads(result)
        return data.get("relevant", False)
    except Exception:
        return False
    
def save_message(uid: str, role: str, content: str, max_session: int = 5):
    """Save short-term chat in Redis, update profile, and maintain evolving summary."""
    key = f"chat:{uid}"
    redis_client.rpush(key, f"{role}: {content}")
    redis_client.ltrim(key, -max_session, -1)

    #relevant = False
    if role == "user":
        # Only update profile if relevant
        if is_relevant_message(content):  
            update_profile(uid, content)
            #relevant = True

    # Update summary every 5 messages, but only if something relevant was said
    redis_client.incr(f"counter:{uid}")
    counter = int(redis_client.get(f"counter:{uid}"))
    if counter % 5 == 0:
            update_summary(uid)


def update_summary(uid: str, last_n: int = 5):
    """Evolving summary for Health Coach: includes both user & AI messages, condenses into factual long-term memory."""
    from main import llm  # Your Groq LLM instance
    key = f"chat:{uid}"
    
    # Get last N messages (both user and AI)
    last_msgs = redis_client.lrange(key, -last_n, -1)
    if not last_msgs:
        return

    # Annotate messages for clarity
    annotated_msgs = []
    for msg in last_msgs:
        if ": " in msg:
            role, content = msg.split(": ", 1)
            annotated_msgs.append(f"{role.upper()}: {content}")
    last_msgs_text = "\n".join(annotated_msgs)

    # Retrieve previous summary from Chroma
    prev_summary_docs = vectorstore.similarity_search(
        "summary", k=1, filter={"tag": f"{uid}_summary"}
    )
    prev_summary = prev_summary_docs[0].page_content if prev_summary_docs else ""

    # Build prompt
    if prev_summary:
        prompt = f"""
You are a friendly, motivating Health Coach. Merge the previous summary with the following recent messages into a clear, factual long-term memory.
Previous Summary:
{prev_summary}

Recent Messages:
{last_msgs_text}

Return a neutral, readable summary suitable for guiding health suggestions, motivational nudges, and frailty routines. 
Do NOT invent new things, summarize only what actually occurred (what user and AI said), dont hallucinate.
Max Length: 2000 words.(dont exceed, when new info keeps coming compare with previous summary and use what is relevant for new summary)
Focus on:
- User goals
- Likes/dislikes related to health and activities
- Relevant behaviors or habits
- Previous AI suggestions for context
- Include only facts explicitly mentioned by the user or by previous AI suggestions.
- Do NOT invent new behaviors, habits, or contexts.
"""
    else:
        prompt = f"""
You are a friendly, motivating Health Coach. Summarize the following recent messages into a long-term memory suitable for guiding health suggestions, nudges, and frailty routines. 
Recent Messages:
{last_msgs_text}

Return a neutral, readable summary. Do NOT invent new advice.
Length: 5-7 sentences.
Focus on:
- User goals
- Likes/dislikes related to health and activities
- Relevant behaviors or habits
- Previous AI suggestions for context
- Include only facts explicitly mentioned by the user or by previous AI suggestions.
- Do NOT invent new behaviors, habits, or contexts.
"""

    try:
        new_summary = llm.invoke(prompt).content.strip()
        if new_summary:
            #vectorstore.delete(where={"tag": f"{uid}_summary"})
            vectorstore.add_texts(
                [new_summary],
                metadatas=[{"tag": f"{uid}_summary"}]
            )
    except Exception as e:
        print(f"Failed to generate summary via LLM: {e}")


def get_memory(uid: str, query: str, short_term_limit: int = 4, long_term_k: int = 2):
    """Retrieve combined context: short-term, profile, and relevant long-term summary."""
    key = f"chat:{uid}"

    # Short-term recent messages
    short_term = redis_client.lrange(key, -short_term_limit, -1) or []

    # Profile snapshot
    profile_context = get_profile(uid)

    # Long-term semantic summary
    long_term = []
    if query and query.strip():
        try:
            long_term_docs = vectorstore.similarity_search(
                query, k=long_term_k, filter={"tag": f"{uid}_summary"}
            )
            long_term = [f"[Past Summary]: {doc.page_content}" for doc in long_term_docs]
        except Exception as e:
            print(f"Chroma search failed: {e}")

    return short_term + ([profile_context] if profile_context else []) + long_term
