import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolpickr.core.tool import ToolDefinition
tools = [

    # 🌍 GENERAL / SEARCH
    ToolDefinition(name="web_search", description="Search the web for information.", parameters={"query": "string"}),
    ToolDefinition(name="news_search", description="Search latest news articles.", parameters={"query": "string"}),
    ToolDefinition(name="image_search", description="Search for images.", parameters={"query": "string"}),
    ToolDefinition(name="video_search", description="Search for videos.", parameters={"query": "string"}),

    # 🌦 WEATHER
    ToolDefinition(name="get_weather", description="Get current weather.", parameters={"location": "string"}),
    ToolDefinition(name="get_weather_forecast", description="Get weather forecast.", parameters={"location": "string", "days": "number"}),

    # 💰 FINANCE
    ToolDefinition(name="get_asset_data", description="Fetch financial data for an asset.", parameters={"asset_type": "string", "symbol": "string", "fields": "list"}),
    ToolDefinition(name="get_asset_news", description="Get news for an asset.", parameters={"asset_type": "string", "symbol": "string"}),
    ToolDefinition(name="convert_currency", description="Convert currency.", parameters={"amount": "number", "from": "string", "to": "string"}),
    ToolDefinition(name="calculate_investment_return", description="Calculate investment return.", parameters={"principal": "number", "rate": "number", "time": "number"}),

    # 🛒 E-COMMERCE
    ToolDefinition(name="search_products", description="Search for products.", parameters={"query": "string"}),
    ToolDefinition(name="get_product_details", description="Get product details.", parameters={"product_id": "string"}),
    ToolDefinition(name="compare_products", description="Compare products.", parameters={"product_ids": "list"}),
    ToolDefinition(name="track_order", description="Track an order.", parameters={"order_id": "string"}),

    # 📍 MAPS & TRAVEL
    ToolDefinition(name="find_places", description="Find nearby places.", parameters={"query": "string", "location": "string"}),
    ToolDefinition(name="get_directions", description="Get directions.", parameters={"from": "string", "to": "string"}),
    ToolDefinition(name="get_distance", description="Calculate distance.", parameters={"from": "string", "to": "string"}),
    ToolDefinition(name="book_flight", description="Book a flight.", parameters={"from": "string", "to": "string", "date": "string"}),
    ToolDefinition(name="search_hotels", description="Search hotels.", parameters={"location": "string", "checkin": "string", "checkout": "string"}),

    # 🍔 FOOD
    ToolDefinition(name="find_restaurants", description="Find restaurants.", parameters={"location": "string", "cuisine": "string"}),
    ToolDefinition(name="get_menu", description="Get restaurant menu.", parameters={"restaurant_name": "string"}),
    ToolDefinition(name="order_food", description="Order food.", parameters={"restaurant": "string", "items": "list"}),

    # 📅 PRODUCTIVITY
    ToolDefinition(name="create_calendar_event", description="Create calendar event.", parameters={"title": "string", "date": "string", "time": "string"}),
    ToolDefinition(name="get_calendar_events", description="Get calendar events.", parameters={"date": "string"}),
    ToolDefinition(name="set_reminder", description="Set reminder.", parameters={"task": "string", "time": "string"}),
    ToolDefinition(name="send_email", description="Send email.", parameters={"to": "string", "subject": "string", "body": "string"}),

    # 📂 FILE SYSTEM
    ToolDefinition(name="read_file", description="Read file.", parameters={"path": "string"}),
    ToolDefinition(name="write_file", description="Write file.", parameters={"path": "string", "content": "string"}),
    ToolDefinition(name="delete_file", description="Delete file.", parameters={"path": "string"}),
    ToolDefinition(name="list_files", description="List files in directory.", parameters={"path": "string"}),

    # 🧮 UTILITIES
    ToolDefinition(name="calculator", description="Evaluate math expression.", parameters={"expression": "string"}),
    ToolDefinition(name="unit_convert", description="Convert units.", parameters={"value": "number", "from": "string", "to": "string"}),
    ToolDefinition(name="get_time", description="Get current time.", parameters={"location": "string"}),

    # 🧠 KNOWLEDGE / AI
    ToolDefinition(name="ask_knowledge_base", description="Query internal knowledge base.", parameters={"question": "string"}),
    ToolDefinition(name="summarize_text", description="Summarize text.", parameters={"text": "string"}),
    ToolDefinition(name="translate_text", description="Translate text.", parameters={"text": "string", "target_language": "string"}),
    ToolDefinition(name="extract_entities", description="Extract entities from text.", parameters={"text": "string"}),

    # 🏥 HEALTH
    ToolDefinition(name="search_symptoms", description="Search symptoms.", parameters={"symptoms": "string"}),
    ToolDefinition(name="find_doctors", description="Find doctors.", parameters={"specialty": "string", "location": "string"}),
    ToolDefinition(name="book_appointment", description="Book doctor appointment.", parameters={"doctor": "string", "date": "string"}),

    # 🎓 EDUCATION
    ToolDefinition(name="search_courses", description="Search online courses.", parameters={"query": "string"}),
    ToolDefinition(name="get_course_details", description="Get course details.", parameters={"course_id": "string"}),
    ToolDefinition(name="recommend_books", description="Recommend books.", parameters={"topic": "string"}),

    # 🎵 MEDIA
    ToolDefinition(name="play_music", description="Play music.", parameters={"song": "string"}),
    ToolDefinition(name="get_song_info", description="Get song info.", parameters={"song": "string"}),
    ToolDefinition(name="recommend_movies", description="Recommend movies.", parameters={"genre": "string"}),
    ToolDefinition(name="get_movie_details", description="Get movie details.", parameters={"movie": "string"}),

    # 🚗 TRANSPORT
    ToolDefinition(name="book_ride", description="Book a ride.", parameters={"pickup": "string", "drop": "string"}),
    ToolDefinition(name="get_ride_estimate", description="Get ride estimate.", parameters={"pickup": "string", "drop": "string"}),

    # 💼 JOBS
    ToolDefinition(name="search_jobs", description="Search jobs.", parameters={"query": "string", "location": "string"}),
    ToolDefinition(name="get_job_details", description="Get job details.", parameters={"job_id": "string"}),

    # 🏠 REAL ESTATE
    ToolDefinition(name="search_properties", description="Search properties.", parameters={"location": "string", "budget": "number"}),
    ToolDefinition(name="get_property_details", description="Get property details.", parameters={"property_id": "string"}),

    # 🔐 SECURITY
    ToolDefinition(name="generate_password", description="Generate secure password.", parameters={"length": "number"}),
    ToolDefinition(name="check_password_strength", description="Check password strength.", parameters={"password": "string"}),

    # 📊 DATA / ANALYTICS
    ToolDefinition(name="run_sql_query", description="Run SQL query.", parameters={"query": "string"}),
    ToolDefinition(name="generate_report", description="Generate report.", parameters={"data": "string"}),

    # 🧾 DOCUMENTS
    ToolDefinition(name="create_document", description="Create document.", parameters={"title": "string", "content": "string"}),
    ToolDefinition(name="edit_document", description="Edit document.", parameters={"doc_id": "string", "content": "string"}),

    # 🧠 ADVANCED TASKS
    ToolDefinition(name="plan_trip", description="Plan a trip itinerary.", parameters={"destination": "string", "days": "number"}),
    ToolDefinition(name="analyze_sentiment", description="Analyze sentiment of text.", parameters={"text": "string"}),
]