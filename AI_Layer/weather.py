import requests
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import os
import json
import redis

# API Configuration - WeatherAPI.com
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "436c7058caf44466979233535250810")
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
AIR_QUALITY_ENABLED = True  # WeatherAPI includes AQI by default

# Redis for caching
REDIS_HOST = "localhost"
REDIS_PORT = 6379
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

CACHE_DURATION = 1800  # 30 minutes in seconds


def get_weather(location: str = "Adelaide, AU", use_cache: bool = True) -> Optional[Dict]:
    """
    Fetch current weather data with caching (WeatherAPI.com version).
    
    Args:
        location: City name or coordinates
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dict with weather info or None if failed
    """
    cache_key = f"weather:{location}"
    
    # Check cache first
    if use_cache:
        cached = redis_client.get(cache_key)
        if cached:
            try:
                cached_data = json.loads(cached)
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cached_data["cached_at"])
                if datetime.utcnow() - cache_time < timedelta(seconds=CACHE_DURATION):
                    return cached_data
            except (json.JSONDecodeError, KeyError):
                pass
    
    # Fetch fresh data
    try:
        # Ensure location is not empty
        if not location or location.strip() == "":
            location = "Adelaide, AU"
        
        params = {
            "key": WEATHER_API_KEY,
            "q": location,
            "aqi": "yes"  # Include air quality
        }
        
        print(f"[DEBUG] Fetching weather for: {location}")  # Debug log
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"[DEBUG] Weather fetch successful")  # Debug log
        
        # Extract relevant info (WeatherAPI.com format)
        weather_info = {
            "temperature": data["current"]["temp_c"],
            "feels_like": data["current"]["feelslike_c"],
            "condition": data["current"]["condition"]["text"],
            "description": data["current"]["condition"]["text"].lower(),
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_kph"] / 3.6,  # Convert to m/s
            "timestamp": datetime.utcnow().isoformat(),
            "location": data["location"]["name"],
            "cached_at": datetime.utcnow().isoformat(),
            "lat": data["location"]["lat"],
            "lon": data["location"]["lon"],
            # Air quality data (WeatherAPI includes this)
            "aqi": data["current"].get("air_quality", {})
        }
        
        # Cache the result
        redis_client.setex(cache_key, CACHE_DURATION, json.dumps(weather_info))
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Weather API error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except KeyError as e:
        print(f"Weather data parsing error: {e}")
        print(f"Response data: {data if 'data' in locals() else 'No data'}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def get_air_quality(lat: float, lon: float) -> Optional[Dict]:
    """
    Get air quality from weather data (WeatherAPI.com includes AQI).
    
    Args:
        lat: Latitude (not used, for compatibility)
        lon: Longitude (not used, for compatibility)
    
    Returns:
        Dict with AQI info or None
    """
    # WeatherAPI.com includes AQI in the main weather response
    # This function exists for API compatibility but isn't needed
    return None


def parse_aqi_from_weather(weather_data: Dict) -> Optional[Dict]:
    """
    Extract AQI from WeatherAPI.com response.
    
    WeatherAPI uses US EPA standard (1-6 scale):
    1 = Good, 2 = Moderate, 3 = Unhealthy for sensitive, 
    4 = Unhealthy, 5 = Very Unhealthy, 6 = Hazardous
    """
    if not weather_data or "aqi" not in weather_data:
        return None
    
    aqi_data = weather_data["aqi"]
    
    # Get US EPA index
    us_epa_index = aqi_data.get("us-epa-index", 1)
    
    aqi_labels = {
        1: "Good",
        2: "Moderate", 
        3: "Unhealthy for Sensitive Groups",
        4: "Unhealthy",
        5: "Very Unhealthy",
        6: "Hazardous"
    }
    
    return {
        "aqi": us_epa_index,
        "label": aqi_labels.get(us_epa_index, "Unknown"),
        "timestamp": datetime.utcnow().isoformat()
    }


def get_time_of_day() -> Tuple[str, int]:
    """
    Get current time period and hour.
    
    Returns:
        Tuple of (period_name, hour)
    """
    now = datetime.now()
    hour = now.hour
    
    if 5 <= hour < 8:
        return "early_morning", hour
    elif 8 <= hour < 12:
        return "morning", hour
    elif 12 <= hour < 17:
        return "afternoon", hour
    elif 17 <= hour < 21:
        return "evening", hour
    else:
        return "night", hour


def format_weather_context(weather: Dict, aqi: Optional[Dict] = None) -> str:
    """
    Format weather data into a readable string for LLM context.
    """
    if not weather:
        return ""
    
    # Parse AQI from weather data if not provided
    if not aqi and "aqi" in weather:
        aqi = parse_aqi_from_weather(weather)
    
    context = f"""
[Current Weather in {weather['location']}]:
- Temperature: {weather['temperature']:.1f}°C (feels like {weather['feels_like']:.1f}°C)
- Conditions: {weather['description']}
- Humidity: {weather['humidity']}%
- Wind Speed: {weather['wind_speed']:.1f} m/s"""
    
    if aqi:
        context += f"\n- Air Quality: {aqi['label']} (AQI: {aqi['aqi']})"
    
    return context


def should_fetch_weather(user_text: str) -> bool:
    """
    Determine if the user query explicitly requires weather information.
    Only returns True if user is clearly asking about weather or activity planning.
    """
    text_lower = user_text.lower().strip()
    
    # Direct weather queries
    direct_weather = [
        "weather", "temperature", "how hot", "how cold", "raining", 
        "sunny", "cloudy", "forecast"
    ]
    
    # Activity planning queries (need weather context)
    activity_planning = [
        "what should i do today", "what can i do today", "what activity",
        "should i go", "can i go", "good day for", "good time for",
        "exercise today", "walk today", "outdoor today", "outside today", "the air quality", "how'ss the air","how's the temperature"
    ]
    
    # Check direct weather queries
    if any(keyword in text_lower for keyword in direct_weather):
        return True
    
    # Check activity planning queries
    if any(phrase in text_lower for phrase in activity_planning):
        return True
    
    # Don't trigger on generic mentions
    # e.g., "I like walking" or "I went to the park" shouldn't trigger weather
    return False


def is_safe_outdoor_time() -> Tuple[bool, str]:
    """
    Check if current time is safe for outdoor activities for elderly.
    """
    period, hour = get_time_of_day()
    
    if period == "night":
        return False, "It's nighttime - outdoor activities aren't safe right now"
    elif period == "early_morning" and hour < 7:
        return False, "It's quite early - wait until sunrise for outdoor activities"
    elif period == "evening" and hour >= 20:
        return False, "It's getting dark - better to stay indoors now"
    
    return True, ""


def analyze_weather_safety(weather: Dict, aqi: Optional[Dict] = None) -> Dict:
    """
    Analyze weather conditions and return safety assessment.
    Returns structured data for AI to interpret naturally.
    """
    if not weather:
        return {"safe": False, "reason": "no_data"}
    
    # Parse AQI from weather data if not provided
    if not aqi and "aqi" in weather:
        aqi = parse_aqi_from_weather(weather)
    
    temp = weather["temperature"]
    condition = weather["condition"].lower()
    wind_speed = weather["wind_speed"]
    
    analysis = {
        "temperature": {
            "value": temp,
            "status": "good" if 15 <= temp <= 28 else "caution" if 10 <= temp <= 35 else "unsafe",
            "concerns": []
        },
        "air_quality": {
            "aqi": aqi["aqi"] if aqi else None,
            "label": aqi["label"] if aqi else "Unknown",
            "status": "good" if (aqi and aqi["aqi"] < 4) else "unsafe" if aqi else "unknown"
        },
        "wind": {
            "speed": wind_speed,
            "status": "good" if wind_speed < 7 else "caution" if wind_speed < 10 else "unsafe"
        },
        "conditions": {
            "description": condition,
            "status": "good" if not any(x in condition for x in ["rain", "storm", "snow", "thunder"]) else "unsafe"
        },
        "time_safety": is_safe_outdoor_time(),
        "overall_safe": True
    }
    
    # Add temperature concerns
    if temp > 35:
        analysis["temperature"]["concerns"].append("extremely hot")
        analysis["overall_safe"] = False
    elif temp > 30:
        analysis["temperature"]["concerns"].append("very hot")
    elif temp < 5:
        analysis["temperature"]["concerns"].append("extremely cold")
        analysis["overall_safe"] = False
    elif temp < 10:
        analysis["temperature"]["concerns"].append("quite cold")
    
    # Overall safety check
    if (analysis["air_quality"]["status"] == "unsafe" or 
        analysis["wind"]["status"] == "unsafe" or 
        analysis["conditions"]["status"] == "unsafe" or
        not analysis["time_safety"][0]):
        analysis["overall_safe"] = False
    
    return analysis


def get_user_location(uid: str) -> str:
    """
    Get user's location from profile, fallback to default.
    """
    try:
        profile_key = f"profile:{uid}"
        profile = redis_client.get(profile_key)
        if profile:
            profile_data = json.loads(profile)
            return profile_data.get("location", "Adelaide, AU")
    except Exception as e:
        print(f"Error getting user location: {e}")
    
    return "Adelaide, AU"


def save_user_location(uid: str, location: str):
    """
    Save user's location to profile.
    """
    try:
        profile_key = f"profile:{uid}"
        profile = redis_client.get(profile_key)
        profile_data = json.loads(profile) if profile else {}
        profile_data["location"] = location
        redis_client.set(profile_key, json.dumps(profile_data))
    except Exception as e:
        print(f"Error saving user location: {e}")