#!/usr/bin/env python3
"""
Simple Travel Agent that works with OpenRouter
This agent interviews users about their trip purpose without using function calling.
"""

import json
from typing import Dict, Any
from litellm import completion, acompletion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_simple_travel_agent(user_input: str) -> Dict[str, Any]:
    """
    Run a simple travel agent that interviews the user about trip purpose
    """
    
    # System prompt for the travel agent
    system_prompt = """You are a travel agent with two main goals:
1. Interview the user about the purpose of their trip
2. Terminate the conversation after gathering the information

When the user tells you about their trip purpose, respond with:
"I understand you. For your request I will look information on the Internet."

Then end the conversation with:
"Thank you for sharing your travel purpose! Terminating..."

Keep your responses concise and focused on these two goals."""

    # Create the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    try:
        # Call the LLM
        response = await acompletion(
            model="openrouter/google/gemini-2.0-flash-exp:free",
            messages=messages,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content
        
        # Create a memory-like structure for consistency
        memory = [
            {"type": "user", "content": user_input},
            {"type": "assistant", "content": result}
        ]
        
        return {
            "memory": memory,
            "status": "completed",
            "response": result
        }
        
    except Exception as e:
        return {
            "memory": [{"type": "error", "content": str(e)}],
            "status": "error",
            "response": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    # Test the simple travel agent
    import asyncio
    user_input = input("What is the purpose of your trip? ")
    result = asyncio.run(run_simple_travel_agent(user_input))
    
    print("\n📝 Travel Agent Memory:")
    print("=" * 50)
    for item in result["memory"]:
        print(f"\n{item['type'].upper()}: {item['content']}")
    print("\n" + "=" * 50)
    #print(f"Status: {result['status']}")

