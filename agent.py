import os
import sys
import time
import threading
import itertools
import yaml
import json
import re
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

try:
    from prompt_toolkit.auto_suggestion import AutoSuggestFromHistory
    AUTO_SUGGEST = AutoSuggestFromHistory()
except ImportError:
    AUTO_SUGGEST = None

class Spinner:
    """A simple animated CLI spinner for tool execution feedback."""
    def __init__(self, msg="Processing"):
        self.msg = msg
        self.running = False
        self.thread = None

    def _spin(self):
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self.running:
            sys.stdout.write(f"\r\033[96m{next(spinner)} {self.msg}...\033[0m\033[K")
            sys.stdout.flush()
            time.sleep(0.08)
        
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join()

with open('/app/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

os.environ["OLLAMA_HOST"] = config['agent']['host']
MODEL = config['agent']['model']
OPTIONS = config['agent']['options']
BASE_SYSTEM_PROMPT = config['agent']['system_prompt']
MAX_TOKENS = OPTIONS.get('num_ctx', 16384)

from ollama import Client
ollama_client = Client(host=config['agent']['host'])

from tools import (
    ALL_TOOLS, 
    AVAILABLE_TOOLS_MAP, 
    get_tools_prompt_summary, 
    init_db, 
    _init_chat_db, 
    _init_checkpoint_db, 
    _load_chat_history_from_db, 
    _save_message_to_db, 
    clear_chat_history
)

init_db()
_init_chat_db()
_init_checkpoint_db()

SHELL_HISTORY_FILE = "/app/memory/.agent_history"

custom_style = Style.from_dict({
    'prompt': 'ansigreen bold',
    'input': 'ansiwhite',
    'completion-menu.completion': 'bg:#000000 #ffffff',
    'completion-menu.completion.current': 'bg:#005f5f #ffffff',
})

session = PromptSession(
    history=FileHistory(SHELL_HISTORY_FILE),
    auto_suggest=AUTO_SUGGEST,
    style=custom_style
)

def build_full_system_prompt() -> str:
    return BASE_SYSTEM_PROMPT + get_tools_prompt_summary()

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def enforce_token_budget(msgs: list, max_tokens: int = MAX_TOKENS) -> list:
    """Optimized token budgeter that truncates overly large tool outputs instead of dropping them."""
    if not msgs:
        return msgs
    
    system_msg = msgs[0]
    other_msgs = msgs[1:]
    
    system_tokens = estimate_tokens(system_msg.get('content', ''))
    reserve_tokens = 4096  # Reserved for model generation
    budget = max_tokens - system_tokens - reserve_tokens
    if budget < 1024:
        budget = 1024  
    
    current_tokens = 0
    retained_msgs = []
    
    # Always keep the most recent user message
    if other_msgs:
        latest_msg = other_msgs[-1]
        retained_msgs.insert(0, latest_msg)
        current_tokens += estimate_tokens(str(latest_msg.get('content', '')))
        
        # Iterate backward through history
        for m in reversed(other_msgs[:-1]):
            content = str(m.get('content', ''))
            t_count = estimate_tokens(content)
            
            if current_tokens + t_count > budget:
                # If we exceed budget but have some room left, gracefully truncate the message
                remaining = budget - current_tokens
                if remaining > 250:
                    char_limit = remaining * 4
                    truncated_content = content[:char_limit] + "\n\n[System: Content truncated to fit context budget...]"
                    m_copy = m.copy()
                    m_copy['content'] = truncated_content
                    retained_msgs.insert(0, m_copy)
                    current_tokens += estimate_tokens(truncated_content)
                break
                
            retained_msgs.insert(0, m)
            current_tokens += t_count
            
    return [system_msg] + retained_msgs

STATIC_SYSTEM_PROMPT = build_full_system_prompt()

try:
    raw_messages = _load_chat_history_from_db(limit=20)
    if raw_messages:
        messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}] + raw_messages
    else:
        messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]
except Exception:
    messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]

def append_and_save_message(msg: dict):
    messages.append(msg)
    _save_message_to_db(msg)

thinking_enabled = True
print(f"Agent initialized with {MODEL} (Thinking: ON). Type 'exit' to quit, '/think [on/off]' to toggle reasoning, or '/forget' to clear chat.")

while True:
    try:
        user_input = session.prompt("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if user_input.lower().startswith('/forget'):
            clear_chat_history()
            messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]
            print("\n[System]: Chat history has been forgotten and cleared.")
            continue
            
        if user_input.lower().startswith('/think'):
            parts = user_input.split()
            if len(parts) > 1:
                arg = parts[1].lower()
                if arg in ['on', 'true', '1']:
                    thinking_enabled = True
                elif arg in ['off', 'false', '0']:
                    thinking_enabled = False
            else:
                thinking_enabled = not thinking_enabled
            print(f"\n[System]: Thinking mode is now {'ENABLED' if thinking_enabled else 'DISABLED'}.")
            continue
            
        append_and_save_message({'role': 'user', 'content': user_input})
        
        while True:
            messages[0]['content'] = STATIC_SYSTEM_PROMPT
            active_messages = enforce_token_budget(messages, MAX_TOKENS)
            
            full_content = ""
            raw_tool_calls = []
            
            in_thinking = False
            in_content = False
            
            # Direct stream iteration for real-time token streaming
            stream = ollama_client.chat(
                model=MODEL,
                messages=active_messages,
                tools=ALL_TOOLS,
                options=OPTIONS,
                think=thinking_enabled,
                stream=True
            )

            for chunk in stream:
                msg_chunk = chunk['message'] if isinstance(chunk, dict) else chunk.message
                
                c_thinking = msg_chunk.get('thinking', '') if isinstance(msg_chunk, dict) else getattr(msg_chunk, 'thinking', '')
                c_content = msg_chunk.get('content', '') if isinstance(msg_chunk, dict) else getattr(msg_chunk, 'content', '')
                c_tools = msg_chunk.get('tool_calls', []) if isinstance(msg_chunk, dict) else getattr(msg_chunk, 'tool_calls', [])
                
                if c_thinking:
                    if not in_thinking:
                        print("\n\033[90m[Thinking Trace]:")
                        in_thinking = True
                    print(c_thinking, end='', flush=True)
                    
                if c_content:
                    if not in_content:
                        if in_thinking:
                            print("\033[0m\n")
                        print("\nAgent: ", end='', flush=True)
                        in_content = True
                    print(c_content, end='', flush=True)
                    full_content += c_content
                    
                # Replace instead of append to prevent stream duplication of arguments
                if c_tools:
                    raw_tool_calls = c_tools
            
            if in_thinking and not in_content:
                print("\033[0m", end='', flush=True)
                
            print() 
            
            final_tool_calls = []
            
            # 1. Parse native tool calls from Ollama API
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    func = tc.get('function', {}) if isinstance(tc, dict) else getattr(tc, 'function', {})
                    t_name = func.get('name', '') if isinstance(func, dict) else getattr(func, 'name', '')
                    t_args = func.get('arguments', {}) if isinstance(func, dict) else getattr(func, 'arguments', {})
                    
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except json.JSONDecodeError:
                            t_args = {}
                            
                    if t_name:
                        final_tool_calls.append({
                            'function': {
                                'name': t_name,
                                'arguments': t_args
                            }
                        })

            # 2. Fallback: Parse raw JSON tool calls in content if native tool calls were empty
            if not final_tool_calls and full_content:
                json_match = re.search(r'(?:<tool_call>|```json)?\s*(\{\s*"name"\s*:\s*"[^"]+".*?\})\s*(?:</tool_call>|```)?', full_content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if 'name' in parsed:
                            final_tool_calls.append({
                                'function': {
                                    'name': parsed['name'],
                                    'arguments': parsed.get('arguments', parsed.get('parameters', {}))
                                }
                            })
                    except json.JSONDecodeError:
                        pass
            
            msg = {'role': 'assistant', 'content': full_content}
            if final_tool_calls:
                msg['tool_calls'] = final_tool_calls
                
            append_and_save_message(msg)
            
            if not final_tool_calls:
                break
                
            print("") 
            for tool_call in msg['tool_calls']:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                with Spinner(f"Executing tool '{func_name}'"):
                    try:
                        tool_res = AVAILABLE_TOOLS_MAP[func_name](**args)
                    except Exception as e:
                        tool_res = f"Error executing tool: {str(e)}"
                
                print(f"  \033[92m[✓]\033[0m System: Finished '{func_name}'")
                
                # Format Tool Output
                res_str = str(tool_res)
                preview = res_str[:250].replace('\n', ' ') + ('...' if len(res_str) > 250 else '')
                print(f"\033[90m  {'─'*50}")
                print(f"  Result: {preview}")
                print(f"  {'─'*50}\033[0m")
                        
                append_and_save_message({
                    'role': 'tool',
                    'name': func_name,
                    'content': res_str
                })
                
    except KeyboardInterrupt:
        print("\nUse 'exit' or 'quit' to close.")
    except EOFError:
        break
