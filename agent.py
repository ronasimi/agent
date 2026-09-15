import os
import sys
import time
import threading
import itertools
import yaml
import json
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
MAX_TOKENS = OPTIONS.get('num_ctx', 8192)

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
    if not msgs:
        return msgs
    
    system_msg = msgs[0]
    other_msgs = msgs[1:]
    
    system_tokens = estimate_tokens(system_msg.get('content', ''))
    budget = max_tokens - system_tokens - 2048
    if budget < 512:
        budget = 512  
    
    current_tokens = 0
    retained_msgs = []
    
    if other_msgs:
        latest_msg = other_msgs[-1]
        retained_msgs.insert(0, latest_msg)
        current_tokens += estimate_tokens(str(latest_msg.get('content', '')))
        other_msgs = other_msgs[:-1]
    
    for m in reversed(other_msgs):
        content = str(m.get('content', ''))
        t_count = estimate_tokens(content)
        if current_tokens + t_count > budget:
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
            tool_call_buffers = [] 
            
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
                        print("\n[Thinking Trace]:")
                        in_thinking = True
                    print(c_thinking, end='', flush=True)
                    
                if c_content:
                    if not in_content:
                        if in_thinking:
                            print("\n")
                        print("\nAgent: ", end='', flush=True)
                        in_content = True
                    print(c_content, end='', flush=True)
                    full_content += c_content
                    
                if c_tools:
                    for tc in c_tools:
                        func = tc.get('function', {}) if isinstance(tc, dict) else getattr(tc, 'function', {})
                        t_name = func.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                        t_args = func.get('arguments', '') if isinstance(tc, dict) else getattr(tc, 'arguments', '')
                        
                        if t_name:
                            tool_call_buffers.append({'name': t_name, 'arguments': t_args or ''})
                        elif t_args and tool_call_buffers:
                            if isinstance(t_args, str):
                                if isinstance(tool_call_buffers[-1]['arguments'], dict):
                                    tool_call_buffers[-1]['arguments'] = json.dumps(tool_call_buffers[-1]['arguments'])
                                tool_call_buffers[-1]['arguments'] += t_args
                            elif isinstance(t_args, dict):
                                tool_call_buffers[-1]['arguments'] = t_args
            
            print() 
            
            final_tool_calls = []
            for buf in tool_call_buffers:
                if buf['name']:
                    parsed_args = {}
                    raw_args = buf['arguments']
                    
                    if isinstance(raw_args, dict):
                        parsed_args = raw_args
                    elif raw_args:
                        try:
                            parsed_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            print(f"  [!] Error parsing arguments for {buf['name']}: {raw_args}")
                            
                    final_tool_calls.append({
                        'function': {
                            'name': buf['name'],
                            'arguments': parsed_args
                        }
                    })
            
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
                
                print(f"  [✓] System: Finished '{func_name}'")
                        
                append_and_save_message({
                    'role': 'tool',
                    'name': func_name,
                    'content': str(tool_res)
                })
                
    except KeyboardInterrupt:
        print("\nUse 'exit' or 'quit' to close.")
    except EOFError:
        break
