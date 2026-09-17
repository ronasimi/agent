# ==========================================
# FILE: tools/research_orchestrator.py
# Smart Research Planning and Execution
# ==========================================
import json
import re
import os
from ollama import Client
import yaml

try:
    with open('/app/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    FAST_MODEL = config['agent'].get('fast_model', 'qwen2.5-coder:1.5b')
except:
    FAST_MODEL = 'qwen2.5-coder:1.5b'

client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def decompose_research_goal(research_goal: str) -> list:
    """
    Use LLM to break down complex research goals into sub-queries.
    
    Args:
        research_goal: High-level research goal
    
    Returns: List of focused search queries
    
    Example:
        >>> queries = decompose_research_goal("AI safety")
        >>> isinstance(queries, list) and len(queries) > 0
        True
    """
    prompt = f"""You are a research planning assistant. Break down the following research goal into 3-5 focused search queries that would collectively provide a comprehensive understanding.

Goal: {research_goal}

Return ONLY a JSON array of strings (queries), e.g.:
["query 1", "query 2", "query 3"]

Each query should be specific and actionable for a search engine."""
    
    try:
        response = client.generate(
            model=FAST_MODEL,
            prompt=prompt,
            options={"temperature": 0.3, "num_ctx": 4096}
        )
        
        # Extract JSON from response
        json_match = re.search(r'\[.*?\]', response['response'], re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
            if isinstance(queries, list):
                return [str(q) for q in queries]
    except Exception as e:
        print(f"  [Debug] Decomposition error: {e}")
    
    # Fallback: return original goal as single query
    return [research_goal]


def orchestrate_research(
    research_goal: str,
    output_format: str = "summary",
    max_queries: int = 3
) -> str:
    """
    Execute a complete research workflow with decomposition, scraping, and synthesis.
    
    Stages:
    1. Decompose goal into sub-queries
    2. Execute searches with deep_research
    3. Synthesize findings with LLM
    4. Generate output in requested format
    
    Args:
        research_goal: High-level research objective
        output_format: Output style (summary, detailed, bullets)
        max_queries: Maximum number of sub-queries
    
    Returns: Synthesized research findings
    
    Example:
        >>> result = orchestrate_research("Latest AI trends")
        >>> isinstance(result, str) and len(result) > 0
        True
    """
    from tools.task_manager import create_task, save_checkpoint, update_task_status, TaskState, log_task
    from tools.deep_research import deep_search_and_scrape, read_research_buffer
    
    # Create tracking task
    task_id = create_task(
        name=f"research_{research_goal[:30].replace(' ', '_')}",
        description=f"Research goal: {research_goal}",
        priority=1,
        state={
            "goal": research_goal,
            "stage": "decomposing",
            "queries": [],
            "completed_queries": []
        }
    )
    
    log_task(f"Starting orchestrated research: {research_goal}", task_id, "info")
    
    # Stage 1: Decompose
    log_task("Decomposing research goal into sub-queries", task_id, "info")
    try:
        queries = decompose_research_goal(research_goal)[:max_queries]
        state = {
            "goal": research_goal,
            "stage": "searching",
            "queries": queries,
            "completed_queries": []
        }
        save_checkpoint(task_id, state)
    except Exception as e:
        log_task(f"Decomposition failed: {e}", task_id, "error")
        update_task_status(task_id, TaskState.FAILED, str(e))
        return f"Research failed during planning: {e}"
    
    # Stage 2: Execute searches
    results_by_query = {}
    for i, query in enumerate(queries):
        try:
            log_task(f"Searching [{i+1}/{len(queries)}]: {query}", task_id, "info")
            result = deep_search_and_scrape(query, max_results=3)
            results_by_query[query] = result
            
            # Checkpoint after each query
            state["completed_queries"].append(query)
            save_checkpoint(task_id, state)
            
        except Exception as e:
            log_task(f"Search failed for query '{query}': {e}", task_id, "warning")
            results_by_query[query] = f"[FAILED] {e}"
    
    # Stage 3: Synthesis
    log_task("Synthesizing findings", task_id, "info")
    
    all_findings = read_research_buffer(research_goal)
    
    synthesis_prompt = f"""Based on the following research findings, provide a {output_format} response to the original goal:

Original Goal: {research_goal}

Findings:
{all_findings}

Provide your response in {output_format} format.
Keep response concise but comprehensive.
Include key takeaways and any important caveats.
If there are conflicting views, mention them."""
    
    try:
        synthesis = client.generate(
            model=FAST_MODEL,
            prompt=synthesis_prompt,
            options={"temperature": 0.5, "num_ctx": 8192}
        )
        output = synthesis['response']
    except Exception as e:
        log_task(f"Synthesis failed: {e}", task_id, "error")
        output = f"Synthesis failed: {e}\n\nRaw findings:\n{all_findings}"
    
    # Mark complete
    update_task_status(task_id, TaskState.COMPLETED)
    log_task("Research orchestration completed", task_id, "info")
    
    return output


def compare_research_findings(goal1: str, goal2: str) -> str:
    """
    Compare research findings on two different topics.
    
    Args:
        goal1: First research goal
        goal2: Second research goal
    
    Returns: Comparison analysis
    """
    findings1 = orchestrate_research(goal1, output_format="bullets")
    findings2 = orchestrate_research(goal2, output_format="bullets")
    
    compare_prompt = f"""Compare these two sets of research findings:

Topic 1: {goal1}
Findings:
{findings1}

Topic 2: {goal2}
Findings:
{findings2}

Provide:
1. Similarities and overlaps
2. Key differences
3. Which topic is more developed
4. Interesting insights from the comparison"""
    
    try:
        response = client.generate(
            model=FAST_MODEL,
            prompt=compare_prompt,
            options={"temperature": 0.5, "num_ctx": 8192}
        )
        return response['response']
    except Exception as e:
        return f"Comparison failed: {e}"


def schedule_research_reminder(research_topic: str, days_ahead: int = 7) -> str:
    """
    Schedule a reminder to re-research a topic (for evolving topics).
    
    Args:
        research_topic: Topic to remind about
        days_ahead: Days until reminder
    
    Returns: Confirmation message
    """
    try:
        from tools.host_tools import create_systemd_timer
        
        timer_name = f"research_{research_topic.replace(' ', '_').lower()}"
        description = f"Re-research: {research_topic}"
        schedule = f"*-*-* 09:00:00"  # Daily at 9 AM
        
        script = f"""#!/bin/bash
echo "Research reminder: {research_topic}" | notify-send -
"""
        
        create_systemd_timer(timer_name, description, schedule, script)
        return f"Research reminder scheduled: '{research_topic}'"
    except Exception as e:
        return f"Error scheduling reminder: {e}"


def batch_research(topics: list, output_format: str = "summary") -> str:
    """
    Execute research on multiple topics.
    
    Args:
        topics: List of research topics
        output_format: Output format for each
    
    Returns: Compiled results for all topics
    """
    results = {}
    
    for topic in topics:
        try:
            result = orchestrate_research(topic, output_format)
            results[topic] = result
        except Exception as e:
            results[topic] = f"Research failed: {e}"
    
    # Compile into single output
    output = "# Batch Research Results\n\n"
    for topic, result in results.items():
        output += f"## {topic}\n\n{result}\n\n---\n\n"
    
    return output


def trending_analysis(base_topic: str) -> str:
    """
    Analyze trending aspects within a topic.
    
    Args:
        base_topic: Main topic to analyze
    
    Returns: Analysis of trending areas
    """
    prompt = f"""What are the most trending and relevant sub-topics within "{base_topic}" right now?
List 5-7 specific trending areas with brief explanations of why they're trending."""
    
    try:
        response = client.generate(
            model=FAST_MODEL,
            prompt=prompt,
            options={"temperature": 0.7, "num_ctx": 4096}
        )
        
        # Parse response to extract topics
        topics = []
        for line in response['response'].split('\n'):
            if line.strip() and not line.startswith('#'):
                topics.append(line.strip())
        
        if topics:
            # Research each trending topic
            results = batch_research(topics[:5], "summary")
            return results
        else:
            return response['response']
    except Exception as e:
        return f"Trending analysis failed: {e}"


def research_with_constraints(
    goal: str,
    constraints: dict = None,
    output_format: str = "summary"
) -> str:
    """
    Research with specific constraints (e.g., date range, language, source type).
    
    Args:
        goal: Research goal
        constraints: Dictionary of constraints:
            - "from_date": "2024-01-01"
            - "to_date": "2025-01-15"
            - "languages": ["en"]
            - "sources": ["academic", "news"]
        output_format: Output format
    
    Returns: Constrained research results
    """
    constraints = constraints or {}
    
    # Enhance goal with constraints
    enhanced_goal = goal
    if "from_date" in constraints:
        enhanced_goal += f" (from {constraints['from_date']})"
    if "sources" in constraints:
        enhanced_goal += f" (sources: {', '.join(constraints['sources'])})"
    
    return orchestrate_research(enhanced_goal, output_format)
