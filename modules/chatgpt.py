from openai import OpenAI
import copy
import time
import json
import sys
import os
from typing import List, Dict, Any, Optional

from modules.token_saver import save_tokens
from modules.helpers import yesno
from modules import gpt_functions
from modules import checklist
from modules import cmd_args
from modules import tokens
from modules import paths

# Initialize OpenAI client
client = OpenAI()


    
# Global state
create_outline = False

def redact_always(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Redact certain messages regardless of content."""
    messages_redact = copy.deepcopy(messages)
    for msg in messages_redact:
        if msg["role"] == "user" and "APPEND_OK" in msg["content"]:
            msg["content"] = "File appended successfully"
            break
    return messages_redact

def redact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Redact sensitive content from messages."""
    messages_redact = copy.deepcopy(messages)
    for msg in messages_redact:
        if msg["role"] == "assistant" and msg["content"] not in [None, "<message redacted>"]:
            msg["content"] = "<message redacted>"
            break
        if msg["role"] == "function" and msg["name"] == "read_file" and msg["content"] not in [None, "<file contents redacted>"]:
            msg["content"] = "<file contents redacted>"
            break
    return messages_redact

def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out certain message types."""
    return [msg for msg in messages if msg["role"] not in ["git"]]

def save_message_history(conv_id: Optional[str], messages: List[Dict[str, Any]]) -> None:
    """Save message history to a file."""
    if conv_id is not None:
        history_file = paths.relative("history", f"{conv_id}.json")
        with open(history_file, "w") as f:
            json.dump(messages, f, indent=4)
def send_message(
    message: Dict[str, Any],
    messages: List[Dict[str, Any]],
    model: str = "gpt-4",
    function_call: Any = "auto",
    retries: int = 0,
    print_message: bool = True,
    conv_id: Optional[str] = None,
    temp: float = 1.0,
) -> List[Dict[str, Any]]:
    """Send a message to OpenAI API with proper error handling and response processing."""
    global create_outline

    def process_tool_calls(tool_calls) -> List[Dict[str, Any]]:
        """Helper function to process tool calls and generate responses."""
        tool_responses = []
        for tool_call in tool_calls:
            try:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                # Get the function from gpt_functions module
                func = getattr(gpt_functions, func_name)
                
                # Handle specific function argument formatting
                if func_name == "make_tasklist":
                    if "tasks" in func_args and isinstance(func_args["tasks"], list):
                        formatted_tasks = []
                        for task in func_args["tasks"]:
                            if isinstance(task, dict):
                                formatted_task = {
                                    "file_involved": task.get("file_involved", "NO_FILE"),
                                    "task_description": task.get("task_description", "")
                                }
                                formatted_tasks.append(formatted_task)
                        func_args["tasks"] = formatted_tasks
                
                # Call the function and get result
                result = func(**func_args)
                
                # Add the tool response message
                tool_responses.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": str(result)
                })
            except Exception as e:
                print(f"Error processing tool call {func_name}: {str(e)}")
                tool_responses.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": f"Error: {str(e)}"
                })
        return tool_responses

    # Process previous tool calls if any exist
    last_message = messages[-1] if messages else None
    if last_message and "tool_calls" in last_message:
        tool_responses = process_tool_calls(last_message["tool_calls"])
        messages.extend(tool_responses)

    # Add user message to message list
    messages.append(message)

    # Handle partial output redaction
    if message.get("content") and "No END_OF_FILE_CONTENT" in message["content"]:
        print("NOTICE: Partial output detected, redacting messages...")
        messages[-2]["content"] = "<file content redacted>"
        messages = redact_messages(messages)

    # Get function definitions based on model
    definitions = copy.deepcopy(gpt_functions.get_definitions(model))

    # Handle task list and checklist logic
    if gpt_functions.active_tasklist or checklist.active_list:
        remove_funcs = ["make_tasklist", "project_finished"]
        definitions = [d for d in definitions if d["name"] not in remove_funcs]
    else:
        definitions = [d for d in definitions if d["name"] != "task_finished"]

    if not gpt_functions.task_operation_performed:
        definitions = [d for d in definitions if d["name"] != "task_finished"]

    # Handle outline creation
    if "no-questions" not in cmd_args.args and gpt_functions.clarification_asked < gpt_functions.initial_question_count:
        definitions = [gpt_functions.ask_clarification_func]
        function_call = {
            "name": "ask_clarification",
            "arguments": "questions"
        }
    elif "no-outline" not in cmd_args.args and not gpt_functions.outline_created:
        print("OUTLINE:  Creating an outline for the project")
        create_outline = True
        definitions = [gpt_functions.ask_clarification_func]
        function_call = "none"
        if not gpt_functions.modify_outline:
            messages.append({
                "role": "user",
                "content": "Please tell me in full detail how you will implement this project. Write it in the first person as if you are the one who will be creating it. Start sentences with 'I will', 'Then I will' and 'Next I will'"
            })
        gpt_functions.outline_created = True

    # Convert functions to tools format for newer API
    tools = [{"type": "function", "function": d} for d in definitions] if definitions else None
    
    print("GPT-API: Waiting... ", end="", flush=True)

    # Save message history
    save_message_history(conv_id, messages)

    response_message = None  # Initialize response_message
    try:
        # Make API call with updated parameters
        response = client.chat.completions.create(
            model=model,
            messages=save_tokens(filter_messages(messages)),
            tools=tools,
            tool_choice="auto" if function_call == "auto" else {
                "type": "function",
                "function": function_call
            } if function_call != "none" else None,
            temperature=temp,
            timeout=120
        )

        # Process tokens and print usage
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        total_tokens = prompt_tokens + completion_tokens

        tokens.add({
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }, model)

        token_cost = round(tokens.get_token_cost(model), 2)
        print(f"OK! (+{total_tokens} tokens, total {int(tokens.token_usage['total'])} / {token_cost} USD)")

        # Process response
        messages = redact_always(messages)
        choice = response.choices[0]
        message_obj = choice.message
        
        # Build response message with proper handling of None content
        response_message = {
            "role": message_obj.role,
            "content": message_obj.content if message_obj.content is not None else ""
        }

        # Handle tool calls
        if message_obj.tool_calls:
            response_message["tool_calls"] = [{
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            } for tool_call in message_obj.tool_calls]

        messages.append(response_message)

        # Handle any tool calls in the current response immediately
        if "tool_calls" in response_message:
            tool_responses = process_tool_calls(response_message["tool_calls"])
            messages.extend(tool_responses)

    except Exception as e:
        # Handle errors
        if retries >= 4:
            raise e

        print(f"\nERROR: OpenAI request failed... {str(e)}")
        print("Trying again in 5 seconds...")
        time.sleep(5)

        messages.pop()
        save_message_history(conv_id, messages)

        return send_message(
            message=message,
            messages=messages,
            model=model,
            function_call=function_call,
            retries=retries+1,
            conv_id=conv_id,
            print_message=print_message,
            temp=temp
        )

    save_message_history(conv_id, messages)

    # Only print message if there is content and print_message is True
    if print_message and response_message and response_message.get("content"):
        print("\n## ChatGPT Responded ##\n```\n")
        print(response_message["content"])
        print("\n```\n")

    return messages

# Export global variables
__all__ = ['create_outline', 'send_message']