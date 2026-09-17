# ==========================================
# FILE: agent.py
# ==========================================
import os
import sys
import time
import threading
import itertools
import yaml
import json
import re
import base64
import requests
import uuid
from io import BytesIO
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

try:
    from prompt_toolkit.auto_suggestion import AutoSuggestFromHistory
    AUTO_SUGGEST = AutoSuggestFromHistory()
except ImportError:
    AUTO_SUGGEST = None

class Spinner:
    def __init__(self, msg="Processing"):
        # Strip newlines and truncate long messages to prevent terminal line wrapping
        clean_msg = str(msg).replace('\n', ' ').replace('\r', ' ')
        if len(clean_msg) > 65:
            clean_msg = clean_msg[:62] + "..."
        self.msg = clean_msg
        self.running = False
        self.thread = None

    def _spin(self):
        spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        while self.running:
            sys.stdout.write(f"\r\033[96m{next(spinner)} {self.msg}\033[0m\033[K")
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
FAST_MODEL = "qwen2.5-coder:1.5b"
OPTIONS = config['agent']['options']
BASE_SYSTEM_PROMPT = config['agent']['system_prompt']
MAX_TOKENS = OPTIONS.get('num_ctx', 32768)
MAX_ITERATIONS = 30

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
    clear_chat_history,
    get_all_memories_prompt_summary
)
from tools.deep_research import deep_search_and_scrape, read_research_buffer, clear_research_buffer
from tools.pdf_generator import generate_pdf_report

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

AUTO_MEMORY_SYSTEM_INSTRUCTION = """
### Automatic Fact Memory Protocol
- You must actively scan conversation turns for unchanging facts (e.g., user preferences, hardware configurations, system paths, IP/subnet assignments, network topology, key relationships).
- Whenever the user introduces or updates an unchanging fact, proactively execute the 'remember(topic, fact)' tool in the background to commit it to long-term memory.
- Do NOT prompt the user for permission before saving a fact—simply invoke the tool seamlessly.

### Native Task Scheduling Protocol
- NEVER use Python background threads or `sleep()` to schedule future tasks, alarms, or reminders.
- You have direct access to the host's systemd user directory at `/app/workspace/host_timers`.
- To schedule a reminder, write a `<name>.service` file and a `<name>.timer` file directly into the `host_timers` directory.
- The `.service` file should use native host commands, e.g., `ExecStart=/usr/bin/notify-send "Reminder" "Message"`.
- After writing both files, use `execute_shell` to run `systemctl --user daemon-reload` followed by `systemctl --user enable --now <name>.timer`.
"""

def build_full_system_prompt() -> str:
    return (
        BASE_SYSTEM_PROMPT 
        + AUTO_MEMORY_SYSTEM_INSTRUCTION 
        + get_tools_prompt_summary() 
        + get_all_memories_prompt_summary()
    )

def estimate_tokens(text: str) -> int:
    if not text: return 0
    words = len(text.split())
    punctuation = len(re.findall(r'[^\w\s]', text))
    return int((words * 1.3) + (punctuation * 0.7))

def enforce_token_budget(msgs: list, max_tokens: int = MAX_TOKENS) -> list:
    if not msgs: return msgs
    system_msg = msgs[0]
    other_msgs = msgs[1:]
    system_tokens = estimate_tokens(system_msg.get('content', ''))
    budget = max(1024, max_tokens - system_tokens - 4096)
    
    current_tokens = 0
    retained_msgs = []
    
    if other_msgs:
        latest_msg = other_msgs[-1]
        latest_content = str(latest_msg.get('content', ''))
        latest_t_count = estimate_tokens(latest_content)
        
        if latest_t_count > budget:
            safe_length = max(100, budget * 3)
            latest_msg = latest_msg.copy()
            latest_msg['content'] = latest_content[:safe_length] + "\n\n[System: Latest message truncated to fit context budget...]"
            latest_t_count = estimate_tokens(str(latest_msg['content']))

        retained_msgs.insert(0, latest_msg)
        current_tokens += latest_t_count
        
        for m in reversed(other_msgs[:-1]):
            content = str(m.get('content', ''))
            t_count = estimate_tokens(content)
            
            if current_tokens + t_count > budget:
                remaining = budget - current_tokens
                if remaining > 250:
                    m_copy = m.copy()
                    is_json_handled = False
                    
                    try:
                        data = json.loads(content)
                        if isinstance(data, list) and len(data) > 0:
                            fraction = max(1, int(len(data) * (remaining * 4 / len(content))))
                            m_copy['content'] = json.dumps(data[:fraction]) + "\n\n[System: JSON array truncated to fit context budget...]"
                            is_json_handled = True
                        elif isinstance(data, dict) and len(data) > 0:
                            keys = list(data.keys())
                            fraction = max(1, int(len(keys) * (remaining * 4 / len(content))))
                            m_copy['content'] = json.dumps({k: data[k] for k in keys[:fraction]}) + "\n\n[System: JSON dict truncated...]"
                            is_json_handled = True
                    except Exception:
                        pass
                        
                    if not is_json_handled:
                        m_copy['content'] = content[:remaining * 4] + "\n\n[System: Content truncated to fit context budget...]"
                    retained_msgs.insert(0, m_copy)
                break
            retained_msgs.insert(0, m)
            current_tokens += t_count
            
    return [system_msg] + retained_msgs

STATIC_SYSTEM_PROMPT = build_full_system_prompt()

try:
    raw_messages = _load_chat_history_from_db(limit=20)
    messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}] + raw_messages if raw_messages else [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]
except Exception:
    messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]

def append_and_save_message(msg: dict):
    messages.append(msg)
    _save_message_to_db(msg)

def encode_image(path_str: str) -> str:
    if path_str.startswith("http://") or path_str.startswith("https://"):
        try:
            resp = requests.get(path_str, headers={"User-Agent": "Agent-Auto-Vision/1.0"}, timeout=10)
            return base64.b64encode(resp.content).decode('utf-8')
        except: return None
            
    candidates = [path_str, os.path.join("/app/workspace", path_str.lstrip('/')), os.path.join("/app/workspace", os.path.basename(path_str))]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                if p.lower().endswith('.pdf'):
                    try:
                        from pdf2image import convert_from_path
                        pages = convert_from_path(p, first_page=1, last_page=1)
                        if pages:
                            buffered = BytesIO()
                            pages[0].save(buffered, format="PNG")
                            return base64.b64encode(buffered.getvalue()).decode('utf-8')
                    except ImportError:
                        print("  \033[91m[!] System: 'pdf2image' is missing. Cannot render PDF.\033[0m")
                    return None
                else:
                    with open(p, "rb") as img_file: 
                        return base64.b64encode(img_file.read()).decode('utf-8')
            except: pass
    return None

IMAGE_REGEX = r'(?:https?://[^\s>\"\']+\.(?:png|jpg|jpeg|webp|pdf)|/?[\w\-\./]+\.(?:png|jpg|jpeg|webp|pdf))'
thinking_enabled = True

def run_deep_research_pipeline(topic: str):
    print(f"\n\033[95m[Deep Research Engine]: Initializing research on '{topic}'...\033[0m")
    clear_research_buffer()

    # PHASE 1: PLANNER
    planner_prompt = (
        f"You are a Lead Research Planner. Topic: '{topic}'.\n"
        "Generate 3 distinct, specific search queries to thoroughly investigate this topic.\n"
        "Return ONLY a JSON array of strings, e.g. [\"query 1\", \"query 2\", \"query 3\"]."
    )
    with Spinner("Planner: Generating research sub-queries"):
        plan_res = ollama_client.generate(model=FAST_MODEL, prompt=planner_prompt)
        try:
            match = re.search(r'\[.*\]', plan_res['response'], re.DOTALL)
            sub_queries = json.loads(match.group(0)) if match else [topic]
        except Exception:
            sub_queries = [f"{topic} key overview", f"{topic} technical details", f"{topic} latest updates"]

    print(f"\033[94m[Planner]: Generated sub-queries:\033[0m")
    for q in sub_queries:
        print(f"  • {q}")

    # PHASE 2: WORKER
    for query in sub_queries:
        with Spinner(f"Worker: Searching & scraping for '{query}'"):
            deep_search_and_scrape(query, max_results=3)

    # PHASE 3: EVALUATOR
    eval_prompt = (
        f"Target Topic: '{topic}'\n"
        f"Current Buffer Findings:\n{read_research_buffer()}\n\n"
        "As a strict Critical Evaluator, you must verify the accuracy and relevance of the gathered data.\n"
        "Check for two things:\n"
        "1. RELEVANCE: Does the data actually discuss the target topic, or is it about something completely different (e.g., bed frames instead of the insect bed bugs)?\n"
        "2. COMPLETENESS: Are there major knowledge gaps?\n\n"
        "Reply EXACTLY with one of the following formats:\n"
        "- If the data is completely irrelevant or wrong: 'IRRELEVANT: <reason>'\n"
        "- If relevant but missing key info: 'GAP: <specific missing topic to search for>'\n"
        "- If relevant and sufficient: 'COMPLETE'"
    )
    with Spinner("Evaluator: Checking coverage, relevance, and knowledge gaps"):
        eval_res = ollama_client.generate(model=FAST_MODEL, prompt=eval_prompt)['response'].strip()

    if eval_res.startswith("IRRELEVANT:"):
        fail_reason = eval_res.replace("IRRELEVANT:", "").strip()
        print(f"\033[91m[Evaluator]: FAIL - Irrelevant data detected. Reason: {fail_reason}\033[0m")
        print("\033[93m[Evaluator]: Flushing corrupted buffer and executing emergency precise scrape...\033[0m")
        clear_research_buffer()
        with Spinner("Worker: Emergency precise scrape"):
            # Add strict contextual words to force the search engine into the right domain
            deep_search_and_scrape(f"factual information about {topic}", max_results=4)
    elif eval_res.startswith("GAP:"):
        gap_query = eval_res.replace("GAP:", "").strip()
        print(f"\033[93m[Evaluator]: Identified gap -> '{gap_query}'. Executing targeted scrape...\033[0m")
        with Spinner("Worker: Addressing research gap"):
            deep_search_and_scrape(gap_query, max_results=2)
    else:
        print(f"\033[92m[Evaluator]: Information coverage and relevance verified complete.\033[0m")

    # PHASE 4: SYNTHESIZER
    print(f"\033[96m[Synthesizer]: Compiling final report using main model...\033[0m\n")
    all_notes = read_research_buffer()
    
    # Strict safeguard instruction to prevent the synthesizer from confidently presenting garbage
    synthesis_prompt = [
        {'role': 'system', 'content': 'You are a Senior Technical Analyst. Compile a detailed, well-structured research report in markdown format with clear section headings, inline images if relevant, and a dedicated References section with source URLs based ONLY on the provided findings. If the provided findings are completely irrelevant to the target topic, DO NOT write a report. Instead, output a clear error stating the data gathering failed due to irrelevance.'},
        {'role': 'user', 'content': f"Topic: {topic}\n\nGathered Research Buffer:\n{all_notes}\n\nWrite the comprehensive research report now:"}
    ]
    
    stream = ollama_client.chat(model=MODEL, messages=synthesis_prompt, stream=True)
    report_content = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        report_content += content
    print("\n")

    # Save Markdown report to message memory
    append_and_save_message({'role': 'assistant', 'content': f"### Deep Research Report: {topic}\n\n{report_content}"})

    # Automatically generate PDF report in workspace
    clean_topic_filename = re.sub(r'[^\w\-_\. ]', '_', topic).replace(' ', '_').lower()
    pdf_filename = f"report_{clean_topic_filename}.pdf"
    
    with Spinner("Generating PDF report"):
        pdf_res = generate_pdf_report(report_content, output_filename=pdf_filename)
        print(f"\033[92m[PDF Generator]: {pdf_res}\033[0m")

print(f"Agent initialized with {MODEL} (Thinking: ON). Type 'exit' to quit, '/research <topic>' for deep research, '/think [on/off]' to toggle, or '/forget' to clear chat.")
print("Use Ctrl+P followed by Ctrl+Q to detach the terminal and let the agent work in the background.")

while True:
    try:
        user_input = session.prompt("\nYou: ").strip()
        if not user_input: continue
        if user_input.lower() in ['exit', 'quit']: break
            
        if user_input.lower().startswith('/forget'):
            clear_chat_history()
            messages = [{'role': 'system', 'content': STATIC_SYSTEM_PROMPT}]
            print("\n[System]: Chat history has been forgotten and cleared.")
            continue

        if user_input.lower().startswith('/research'):
            topic = user_input[10:].strip()
            if not topic:
                print("[System]: Please provide a topic, e.g., '/research Quantum Computing developments'")
                continue
            run_deep_research_pipeline(topic)
            continue
            
        if user_input.lower().startswith('/think'):
            parts = user_input.split()
            if len(parts) > 1: thinking_enabled = (parts[1].lower() in ['on', 'true', '1'])
            else: thinking_enabled = not thinking_enabled
            print(f"\n[System]: Thinking mode is now {'ENABLED' if thinking_enabled else 'DISABLED'}.")
            continue
            
        if user_input.lower() == '/tools':
            print(f"\n[System]: Active Tools Inventory:\n{get_tools_prompt_summary()}")
            continue

        if user_input.lower() == '/reload':
            from tools.tool_manager import reload_tools
            print(f"\n[System]: {reload_tools()}")
            STATIC_SYSTEM_PROMPT = build_full_system_prompt()
            messages[0]['content'] = STATIC_SYSTEM_PROMPT
            print("[System]: Reloaded knowledge base into active system prompt.")
            continue
            
        if user_input.lower() == '/tasks':
            import sqlite3
            try:
                with sqlite3.connect("/app/memory/knowledge.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT task_name, status, timestamp FROM background_tasks")
                    tasks = cursor.fetchall()
                    if tasks:
                        print("\n[System]: Background Tasks:")
                        for t in tasks:
                            print(f"  - {t[0]} | Status: {t[1]} | Started: {t[2]}")
                    else:
                        print("\n[System]: No background tasks recorded.")
            except Exception as e:
                print(f"\n[System]: Error reading tasks: {e}")
            continue

        msg = {'role': 'user', 'content': user_input}
        detected_images = [encode_image(p) for p in re.findall(IMAGE_REGEX, user_input, re.IGNORECASE) if encode_image(p)]
        if detected_images:
            msg['images'] = detected_images
            print(f"  \033[92m[System]: Attached {len(detected_images)} media file(s).\033[0m")
            
        append_and_save_message(msg)
        
        iterations = 0
        while iterations < MAX_ITERATIONS:
            iterations += 1
            messages[0]['content'] = STATIC_SYSTEM_PROMPT
            active_messages = enforce_token_budget(messages, MAX_TOKENS)
            
            full_content = ""
            full_thinking = ""
            raw_tool_calls = []
            in_thinking, in_content = False, False
            
            stream = ollama_client.chat(model=MODEL, messages=active_messages, tools=ALL_TOOLS, options=OPTIONS, think=thinking_enabled, stream=True)

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
                    full_thinking += c_thinking
                    
                if c_content:
                    if not in_content:
                        if in_thinking: print("\033[0m\n")
                        print("\nAgent: ", end='', flush=True)
                        in_content = True
                    print(c_content, end='', flush=True)
                    full_content += c_content
                    
                if c_tools: raw_tool_calls = c_tools
            
            if in_thinking and not in_content: print("\033[0m", end='', flush=True)
            print() 
            
            final_tool_calls = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    func = tc.get('function', {}) if isinstance(tc, dict) else getattr(tc, 'function', {})
                    t_name = func.get('name', '') if isinstance(func, dict) else getattr(func, 'name', '')
                    t_args = func.get('arguments', {}) if isinstance(func, dict) else getattr(func, 'arguments', {})
                    if isinstance(t_args, str):
                        try: t_args = json.loads(t_args)
                        except json.JSONDecodeError: t_args = {}
                    if t_name in AVAILABLE_TOOLS_MAP:
                        final_tool_calls.append({'id': tc.get('id', uuid.uuid4().hex), 'type': 'function', 'function': {'name': t_name, 'arguments': t_args}})

            # Fallback JSON scanner: Checks full content first, then full thinking trace
            search_target = full_content if full_content else full_thinking
            if not final_tool_calls and search_target:
                json_match = re.search(r'(?:<tool_call>|```json)?\s*(\{\s*"name"\s*:\s*"[^"]+".*?\})\s*(?:</tool_call>|```)?', search_target, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if parsed.get('name') in AVAILABLE_TOOLS_MAP:
                            final_tool_calls.append({'id': uuid.uuid4().hex, 'type': 'function', 'function': {'name': parsed['name'], 'arguments': parsed.get('arguments', parsed.get('parameters', {}))}})
                    except json.JSONDecodeError: pass
            
            asst_msg = {'role': 'assistant', 'content': full_content}
            if final_tool_calls: asst_msg['tool_calls'] = final_tool_calls
            append_and_save_message(asst_msg)
            
            if in_thinking and not full_content and not final_tool_calls:
                print("  \033[93m[System]: Model hallucinated a stop. Prompting to continue...\033[0m")
                append_and_save_message({'role': 'user', 'content': 'You generated a thought but provided no final output. Please present the information or execute the intended tool call.'})
                continue
            
            if not final_tool_calls: break
                
            print("") 
            for tool_call in asst_msg['tool_calls']:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                with Spinner(f"Executing tool '{func_name}'"):
                    try: tool_res = AVAILABLE_TOOLS_MAP[func_name](**args)
                    except Exception as e: tool_res = f"Error executing tool: {str(e)}"
                
                print(f"  \033[92m[✓]\033[0m System: Finished '{func_name}'")
                res_str = str(tool_res)
                
                # Safeguard: Hard limit output size to prevent oversized tool outputs
                if len(res_str) > 25000:
                    res_str = res_str[:25000] + "\n\n[System: Tool output truncated. Returned too much raw data to process directly.]"

                preview = res_str[:250].replace('\n', ' ') + ('...' if len(res_str) > 250 else '')
                print(f"\033[90m  {'─'*50}\n  Result: {preview}\n  {'─'*50}\033[0m")
                
                tool_msg = {'role': 'tool', 'name': func_name, 'content': res_str, 'tool_call_id': tool_call.get('id', uuid.uuid4().hex)}
                tool_images = [encode_image(p) for p in re.findall(IMAGE_REGEX, res_str, re.IGNORECASE) if encode_image(p)]
                if tool_images:
                    tool_msg['images'] = tool_images
                    print(f"  \033[92m[System]: Attached {len(tool_images)} media file(s) from tool output.\033[0m")
                        
                append_and_save_message(tool_msg)
                
        if iterations >= MAX_ITERATIONS:
            print("  \033[91m[!] System: Reached maximum iteration limit. Terminating loop to prevent infinite execution.\033[0m")
            append_and_save_message({'role': 'system', 'content': 'System halted execution: maximum iterations reached. Please summarize your findings so far and call the desktop notification tool.'})
                
    except KeyboardInterrupt:
        print("\nUse 'exit' or 'quit' to close. Use Ctrl+P, Ctrl+Q to detach.")
    except EOFError:
        break
