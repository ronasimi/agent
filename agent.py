import os
import sys
import time
import threading
import itertools
import yaml
import json
import atexit
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.shell import BashLexer

try:
    from prompt_toolkit.auto_suggestion import AutoSuggestFromHistory
    AUTO_SUGGEST = AutoSuggestFromHistory()
except ImportError:
    AUTO_SUGGEST = None

class Spinner:
    """A simple animated CLI spinner to provide immediate background feedback."""
    def __init__(self, msg="Processing"):
        self.msg = msg
        self.running = False
        self.thread = None

    def _spin(self):
        # Braille patterns for a smooth circular spinner
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self.running:
            # \033[96m is Cyan, \033[0m resets, \033[K clears the line to prevent artifacts
            sys.stdout.write(f"\r\033[96m{next(spinner)} {self.msg}...\033[0m\033[K")
            sys.stdout.flush()
            time.sleep(0.08)
        
        # Clear the spinner line completely when done
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

from tools import ALL_TOOLS, AVAILABLE_TOOLS_MAP, get_tools_prompt_summary, init_db

init_db()

SHELL_HISTORY_FILE = "/app/memory/.agent_history"
CHAT_HISTORY_FILE = "/app/memory/chat_history.json"

custom_style = Style.from_dict({
    'prompt': 'ansigreen bold',
    'input': 'ansiwhite',
    'completion-menu.completion': 'bg:#000000 #ffffff',
    'completion-menu.completion.current': 'bg:#005f5f #ffffff',
})

session = PromptSession(
    history=FileHistory(SHELL_HISTORY_FILE),
    auto_suggest=AUTO_SUGGEST,
    lexer=PygmentsLexer(BashLexer),
    style=custom_style
)

def build_full_system_prompt() -> str:
    return BASE_SYSTEM_PROMPT + get_tools_prompt_summary()

def estimate_tokens(text: str) -> int:
    """Fast python-side token estimation (~4 characters per token)."""
    return len(text) // 4

def enforce_token_budget(msgs: list, max_tokens: int = MAX_TOKENS) -> list:
    """Trim older message history from the front while preserving system prompt and recent turns."""
    if not msgs:
        return msgs
    
    system_msg = msgs[0]
    other_msgs = msgs[1:]
    
    system_tokens = estimate_tokens(system_msg.get('content', ''))
    budget = max_tokens - system_tokens - 1000
    
    current_tokens = 0
    retained_msgs = []
    
    for m in reversed(other_msgs):
        content = str(m.get('content', ''))
        t_count = estimate_tokens(content)
        if current_tokens + t_count > budget:
            break
        retained_msgs.insert(0, m)
        current_tokens += t_count
        
    return [system_msg] + retained_msgs

# Load persistent chat history or initialize new
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            raw_messages = json.load(f)
            messages = []
            for m in raw_messages:
                messages.append(m)
            if messages:
                messages[0] = {'role': 'system', 'content': build_full_system_prompt()}
            else:
                messages = [{'role': 'system', 'content': build_full_system_prompt()}]
    except Exception:
        messages = [{'role': 'system', 'content': build_full_system_prompt()}]
else:
    messages = [{'role': 'system', 'content': build_full_system_prompt()}]

def save_chat_history():
    serializable_messages = []
    for m in messages:
        if hasattr(m, 'model_dump'):
            serializable_messages.append(m.model_dump())
        elif hasattr(m, 'dict'):
            serializable_messages.append(m.dict())
        elif isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append(dict(m) if hasattr(m, 'keys') else {"role": getattr(m, 'role', 'assistant'), "content": str(getattr(m, 'content', m))})
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(serializable_messages, f, indent=2)

atexit.register(save_chat_history)

thinking_enabled = True
print(f"Agent initialized with {MODEL} (Thinking: ON). Type 'exit' to quit, or '/think [on/off]' to toggle reasoning.")

while True:
    try:
        user_input = session.prompt("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break
            
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
            
        messages.append({'role': 'user', 'content': user_input})
        
        while True:
            messages[0]['content'] = build_full_system_prompt()
            active_messages = enforce_token_budget(messages, MAX_TOKENS)
            
            # 1. Immediate visual feedback via Streaming
            stream = ollama_client.chat(
                model=MODEL,
                messages=active_messages,
                tools=ALL_TOOLS,
                options=OPTIONS,
                think=thinking_enabled,
                stream=True
            )
            
            full_content = ""
            tool_calls = []
            
            in_thinking = False
            in_content = False
            
            for chunk in stream:
                # Handle variations in Ollama Python SDK object returns
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
                    tool_calls.extend(c_tools)
            
            print() # Newline after the stream completes
            
            # Reconstruct the message object so the tool execution loop still works
            msg = {'role': 'assistant', 'content': full_content}
            if tool_calls:
                msg['tool_calls'] = tool_calls
                
            messages.append(msg)
            
            if not msg.get('tool_calls'):
                break
                
            print("") 
            for tool_call in msg['tool_calls']:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                # 2. Immediate visual feedback for Tool Execution (great for long tools like nmap)
                with Spinner(f"Executing tool '{func_name}'"):
                    try:
                        tool_res = AVAILABLE_TOOLS_MAP[func_name](**args)
                    except Exception as e:
                        tool_res = f"Error executing tool: {str(e)}"
                
                # Clean completion acknowledgment
                print(f"  [✓] System: Finished '{func_name}'")
                        
                messages.append({
                    'role': 'tool',
                    'name': func_name,
                    'content': str(tool_res)
                })
                
    except KeyboardInterrupt:
        print("\nUse 'exit' or 'quit' to save history and close.")
    except EOFError:
        break
