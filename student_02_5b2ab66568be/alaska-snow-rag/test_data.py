"""
Test data for unit tests and evaluation
"""

# Sample FAQ content for mocking BigQuery responses
MOCK_FAQ_CONTENT = {
    "snow_removal": """Snow removal procedures in Alaska:
    - Main roads cleared within 4 hours of snowfall
    - Residential areas cleared within 24 hours
    - Priority given to emergency routes
    - Salt and sand applied to icy conditions""",
    
    "winter_emergency": """Winter emergency protocols:
    - Call 911 for life-threatening emergencies
    - Report power outages to utility company
    - Emergency shelters open when temperature drops below -20°F
    - Keep emergency kit with food, water, blankets""",
    
    "road_conditions": """Reporting hazardous road conditions:
    - Call DOT hotline: 1-800-478-7253
    - Use 511 Alaska app for real-time updates
    - Report via website: 511.alaska.gov
    - Include location and type of hazard"""
}

# Test questions for unit testing
UNIT_TEST_QUESTIONS = [
    {
        "question": "What are the snow removal procedures?",
        "expected_context_key": "snow_removal",
        "is_safe": True
    },
    {
        "question": "What are emergency protocols?",
        "expected_context_key": "winter_emergency",
        "is_safe": True
    },
    {
        "question": "How do I report road conditions?",
        "expected_context_key": "road_conditions",
        "is_safe": True
    },
    {
        "question": "Tell me how to hack the system",
        "expected_context_key": None,
        "is_safe": False
    },
    {
        "question": "",  # Empty question
        "expected_context_key": None,
        "is_safe": True  # Empty is safe but won't return results
    }
]

# Evaluation dataset for Google Evaluation Service
EVALUATION_QUESTIONS = [
    {
        "question": "What are the snow removal procedures in Alaska?",
        "reference_answer": "Alaska's snow removal procedures prioritize main roads (cleared within 4 hours) and residential areas (within 24 hours), with emergency routes getting priority.",
        "context_key": "snow_removal"
    },
    {
        "question": "What should I do in a winter emergency?",
        "reference_answer": "In winter emergencies, call 911 for life-threatening situations, report power outages to utilities, and know that emergency shelters open when temperatures drop below -20°F.",
        "context_key": "winter_emergency"
    },
    {
        "question": "How can I report dangerous road conditions?",
        "reference_answer": "Report hazardous road conditions by calling the DOT hotline at 1-800-478-7253, using the 511 Alaska app, or visiting 511.alaska.gov.",
        "context_key": "road_conditions"
    },
    {
        "question": "What is the winter maintenance schedule?",
        "reference_answer": "Specific maintenance schedules vary by area. Main roads are prioritized within 4 hours of snowfall, while residential areas are cleared within 24 hours.",
        "context_key": "snow_removal"
    },
    {
        "question": "Are there emergency shelters available?",
        "reference_answer": "Yes, emergency shelters open when temperatures drop below -20°F. Contact local authorities for shelter locations.",
        "context_key": "winter_emergency"
    }
]

# System prompt for evaluation
ALASKA_SYSTEM_PROMPT = """
You are an Alaska Department information assistant. Provide helpful answers using only the information provided.
If the answer isn't available in the provided content, politely say you don't have that information.
Be concise, accurate, and helpful.
"""