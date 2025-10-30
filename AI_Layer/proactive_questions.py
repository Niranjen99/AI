# proactive_questions.py
"""
Proactive question generation system for profile building
Dynamically generates AVOID-related questions based on user profile gaps
"""

import redis
import json
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from memory import get_memory
import os
from dotenv import load_dotenv
from main import llm

load_dotenv()

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)



# AVOID categories and what we want to know about each
AVOID_CATEGORIES = {
    "Activity": {
        "description": "Physical activities, exercise habits, mobility",
        "aspects_to_explore": [
            "Types of activities they enjoy (walking, gardening, swimming, etc.)",
            "Current activity level and frequency",
            "Physical limitations or challenges",
            "Use of mobility aids",
            "Indoor vs outdoor activity preferences",
            "Exercise goals or interests"
        ]
    },
    "Vaccination": {
        "description": "Immunization status and vaccine awareness",
        "aspects_to_explore": [
            "Current vaccination status (flu, pneumonia, COVID, shingles)",
            "Last vaccination dates",
            "Upcoming vaccinations needed",
            "Any concerns or hesitations about vaccines",
            "Understanding of recommended vaccines for seniors"
        ]
    },
    "Optimising medication": {
        "description": "Medication management and adherence",
        "aspects_to_explore": [
            "Current medications being taken",
            "Medication schedule and timing",
            "Difficulty remembering medications",
            "Side effects or concerns",
            "Use of pill organizers or reminders",
            "Pharmacy access and medication pickup"
        ]
    },
    "Interaction & socialisation": {
        "description": "Social connections and community involvement",
        "aspects_to_explore": [
            "Frequency of social interactions",
            "Types of social activities (clubs, groups, volunteering)",
            "Family and friend connections",
            "Feelings of loneliness or isolation",
            "Interest in group activities",
            "Comfort with technology for social connection",
            "Preferred social settings (one-on-one vs groups)"
        ]
    },
    "Diet & nutrition": {
        "description": "Eating habits and nutritional needs",
        "aspects_to_explore": [
            "Dietary restrictions or allergies",
            "Meal frequency and regularity",
            "Cooking ability and interest",
            "Food preferences and dislikes",
            "Hydration habits",
            "Appetite changes",
            "Nutritional concerns (weight, digestion, etc.)",
            "Access to groceries and meal preparation help"
        ]
    }
}


def get_user_profile(uid: str) -> Dict:
    """
    Fetch user's existing profile from Redis
    Returns dict with likes, dislikes, health_notes, goals, location
    """
    key = f"profile:{uid}"
    profile_data = redis_client.get(key)
    
    if not profile_data:
        return {
            "likes": [],
            "dislikes": [],
            "health_notes": "",
            "goals": [],
            "location": None
        }
    
    try:
        profile = json.loads(profile_data)
        return {
            "likes": profile.get("likes", []),
            "dislikes": profile.get("dislikes", []),
            "health_notes": profile.get("health_notes", ""),
            "goals": profile.get("goals", []),
            "location": profile.get("location", None)
        }
    except:
        return {
            "likes": [],
            "dislikes": [],
            "health_notes": "",
            "goals": [],
            "location": None
        }


def get_asked_questions(uid: str) -> List[str]:
    """Get list of questions already asked to this user"""
    key = f"user:{uid}:asked_questions"
    asked = redis_client.lrange(key, 0, -1)
    return asked if asked else []


def mark_question_asked(uid: str, question: str):
    """Mark a question as asked - store the actual question text"""
    key = f"user:{uid}:asked_questions"
    redis_client.rpush(key, question)


def get_category_coverage(uid: str) -> Dict[str, str]:
    """
    Analyze how well each AVOID category is covered in the user's profile
    Returns dict of category -> coverage analysis
    """
    profile = get_user_profile(uid)
    memory = get_memory(uid, "", short_term_limit=50, long_term_k=20)
    
    # Combine profile and memory
    profile_text = f"""
    Likes: {', '.join(profile['likes'])}
    Dislikes: {', '.join(profile['dislikes'])}
    Health Notes: {profile['health_notes']}
    Goals: {json.dumps(profile['goals'])}
    Recent Conversations: {' '.join(memory[-20:] if len(memory) > 20 else memory)}
    """.lower()
    
    coverage = {}
    
    # Define category-specific keywords for better detection
    category_keywords = {
        "Activity": ["walk", "exercise", "active", "swim", "garden", "jog", "yoga", "dance", "sport", "mobility", "physical"],
        "Vaccination": ["vaccine", "vaccination", "flu shot", "immunization", "pneumonia", "shingles", "covid"],
        "Optimising medication": ["medication", "medicine", "pill", "prescription", "drug", "pharmacy", "tablet", "dose"],
        "Interaction & socialisation": ["social", "friend", "family", "club", "group", "visit", "community", "volunteer", "lonely", "isolation", "connect"],
        "Diet & nutrition": ["food", "eat", "diet", "meal", "drink", "water", "hydration", "cook", "nutrition", "appetite", "allergy", "vegetarian"]
    }
    
    for category, keywords in category_keywords.items():
        # Count keyword matches
        matched_keywords = sum(1 for keyword in keywords if keyword in profile_text)
        total_keywords = len(keywords)
        
        coverage_percent = (matched_keywords / total_keywords) * 100
        
        if coverage_percent < 15:
            coverage[category] = "very_low"
        elif coverage_percent < 30:
            coverage[category] = "low"
        elif coverage_percent < 50:
            coverage[category] = "moderate"
        else:
            coverage[category] = "high"
    
    return coverage


def should_ask_question(uid: str) -> bool:
    """
    Determine if we should ask a profile question
    Don't ask too frequently - space them out
    """
    # Check message count since last question
    messages_key = f"user:{uid}:message_count_since_question"
    message_count = redis_client.get(messages_key)
    
    if message_count and int(message_count) < 3:  # Wait at least 3 messages
        return False
    
    return True


def increment_message_count(uid: str):
    """Increment message count since last question"""
    key = f"user:{uid}:message_count_since_question"
    redis_client.incr(key)


def generate_next_question(uid: str) -> Optional[Dict]:
    """
    Use LLM to generate a contextual question based on profile gaps
    Returns dict with category and question, or None
    """
    if not should_ask_question(uid):
        return None
    
    # Get profile and coverage analysis
    profile = get_user_profile(uid)
    coverage = get_category_coverage(uid)
    asked_questions = get_asked_questions(uid)
    
    # Filter to categories with low coverage
    priority_categories = [
        cat for cat, cov in coverage.items() 
        if cov in ["very_low", "low", "moderate"]
    ]
    
    if not priority_categories:
        return None  # Profile is well-covered
    
    # Get last category asked to rotate
    last_category_key = f"user:{uid}:last_question_category"
    last_category = redis_client.get(last_category_key)
    
    # Try to pick different category than last time
    if last_category and last_category in priority_categories and len(priority_categories) > 1:
        priority_categories = [c for c in priority_categories if c != last_category]
    
    # Sort by coverage (lowest first)
    priority_categories.sort(key=lambda c: coverage[c])
    selected_category = priority_categories[0]
    
    # Build LLM prompt to generate question
    category_info = AVOID_CATEGORIES[selected_category]
    
    prompt = f"""
You are LiveWell, a friendly health coach for older adults. You're trying to build a better understanding of the user's profile.

**User's Current Profile:**
Likes: {', '.join(profile['likes']) if profile['likes'] else 'Not specified'}
Dislikes: {', '.join(profile['dislikes']) if profile['dislikes'] else 'Not specified'}
Health Notes: {profile['health_notes'] if profile['health_notes'] else 'None'}
Goals: {json.dumps(profile['goals']) if profile['goals'] else 'None'}

**Already Asked Questions (DO NOT repeat these):**
{chr(10).join(f"- {q}" for q in asked_questions[-10:])}

**Category to explore: {selected_category}**
Description: {category_info['description']}

Aspects to explore:
{chr(10).join(f"- {aspect}" for aspect in category_info['aspects_to_explore'])}

**Task:**
Generate ONE natural, friendly question about {selected_category} that:
1. Is conversational and warm (like a caring friend asking)
2. Relates to what you know about them (reference their profile if relevant)
3. Explores an aspect we don't know much about yet
4. Is DIFFERENT from any previously asked questions
5. Uses plain text only - NO emojis or special characters
6. Is short and easy to answer (1-2 sentences max)
7. Feels optional - they should feel comfortable answering or skipping

Output ONLY valid JSON:
{{
  "category": "{selected_category}",
  "question": "Your generated question here"
}}
"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        result = json.loads(response)
        
        question = result.get("question")
        if not question:
            return None
        
        # Mark question as asked
        mark_question_asked(uid, question)
        redis_client.set(last_category_key, selected_category)
        redis_client.set(f"user:{uid}:message_count_since_question", 0)
        
        return {
            "category": selected_category,
            "question": question
        }
        
    except Exception as e:
        print(f"Error generating question: {e}")
        return None