import ollama

def list_local_models() -> str:
    """List all locally available Ollama models. Use this to select an appropriate model before delegating tasks."""
    try:
        response = ollama.list()
        models = [m.get('model', m.get('name', 'unknown')) for m in response.get('models', [])]
        
        if not models:
            return "No local models found."
            
        return "Available local models:\n" + "\n".join(f"- {m}" for m in models)
    except Exception as e:
        return f"Error listing models: {str(e)}"

def delegate_task(task_description: str, model_name: str, temperature: float = 0.2) -> str:
    """Delegate a sub-task to an autonomous sub-agent using a specific local model.
    The sub-agent will run independently until the task is complete, and return a summary of its actions.
    
    Args:
        task_description: The prompt/instructions for the sub-agent.
        model_name: The exact name of the local model to use (use list_local_models first).
        temperature: The temperature parameter for the sub-agent (0.0 to 1.0).
    """
    # Defer import to avoid circular dependency during dynamic module loading
    from tools import ALL_TOOLS, AVAILABLE_TOOLS_MAP

    # Give the sub-agent access to everything EXCEPT the ability to delegate 
    # (prevents infinite recursive agent loops)
    sub_tools = [t for t in ALL_TOOLS if t.__name__ != 'delegate_task']
    sub_tools_map = {name: func for name, func in AVAILABLE_TOOLS_MAP.items() if name != 'delegate_task'}

    system_prompt = (
        "You are an autonomous sub-agent delegated to complete a specific task. "
        "You have access to the system's tools. Use them to solve the problem. "
        "When finished, summarize your findings or actions clearly.\n\n"
        f"YOUR ASSIGNED TASK: {task_description}"
    )

    messages = [{'role': 'system', 'content': system_prompt}]
    output_log = []
    
    # Cap the sub-agent at 15 turns so it doesn't run forever if it gets stuck
    max_turns = 15 
    
    try:
        for turn in range(max_turns):
            response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=sub_tools,
                options={'temperature': temperature}
            )
            
            msg = response['message']
            messages.append(msg)
            
            if msg.get('content'):
                output_log.append(f"[Sub-Agent {model_name}]: {msg.get('content')}")
                
            if not msg.get('tool_calls'):
                break
                
            for tool_call in msg['tool_calls']:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                output_log.append(f"  -> Sub-Agent executing: {func_name}")
                
                try:
                    tool_res = sub_tools_map[func_name](**args)
                except Exception as e:
                    tool_res = f"Error executing tool: {str(e)}"
                    
                messages.append({
                    'role': 'tool',
                    'name': func_name,
                    'content': str(tool_res)
                })
        else:
            output_log.append(f"[System]: Sub-agent terminated after reaching turn limit ({max_turns}).")
            
        return "\n".join(output_log)
    except Exception as e:
        return f"Delegation failed: {str(e)}"
