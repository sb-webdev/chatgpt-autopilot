from openai import OpenAI
import json
from typing import Tuple, List, Dict, Any
from modules import tokens

client = OpenAI()

def make_better(prompt: str, model: str, temp: float = 1.0, messages: List[Dict[str, str]] = []) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Improve a prompt using OpenAI's API.
    
    Args:
        prompt: The original prompt to improve
        model: The OpenAI model to use
        temp: Temperature for response generation
        messages: Existing message history, if any
    
    Returns:
        Tuple of (improved prompt, updated message history)
    """
    if len(prompt.split()) < 80:
        words = "an 80 word"
    else:
        words = "a more"

    if not messages:
        messages = [
            {
                "role": "system",
                "content": "You are a prompt designer for an AI agent that can read and write files from the filesystem and run commands on the computer. The AI agent is used to create all kinds of projects, including programming and content creaton. Please note that the agent can not run GUI applications or run tests. Only describe the project, not how it should be implemented. The prompt will be given to the AI agent as a description of the project to accomplish."
            },
            {
                "role": "user",
                "content": f"Convert this prompt into {words} detailed prompt:\n{prompt}"
            }
        ]
    else:
        messages.append({
            "role": "user",
            "content": f"Please make the following changes to the prompt: {prompt}\n\nRespond with the complete, modified version of the prompt."
        })

    # Define the function/tool for the API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "give_prompt",
                "description": "Give the user the better version of the prompt, in full, including modifications",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Better version of the prompt, in full, including modifications. Can include newlines.",
                        },
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "give_prompt"}
            },
            timeout=60
        )

        # Update token usage
        tokens.add({
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }, model)

        # Get the message from the response
        message = response.choices[0].message
        messages.append({
            "role": message.role,
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        # Extract the improved prompt from the tool call
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            return args["prompt"], messages
        else:
            raise ValueError("No tool call in response")

    except Exception as e:
        print(f"Error improving prompt: {str(e)}")
        return prompt, messages  # Return original prompt if something goes wrong