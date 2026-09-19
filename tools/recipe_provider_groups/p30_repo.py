"""Primitive compatibility recipes for repository and observation tools."""
from __future__ import annotations

P = lambda default, description: {"default": default, "description": description}

RECIPE_SPECS = [
    {
        "key":"compat.repo_status","version":1,"name":"compat.repo_status","target_tool":"repo_status",
        "description":"Read repository status through the focused git_status primitive.","tags":["compat","repo","git","status"],"parameters":{},
        "pipeline":[{"id":"result","tool":"git_status","args":{}}],
    },
    {
        "key":"compat.repo_diff","version":1,"name":"compat.repo_diff","target_tool":"repo_diff",
        "description":"Read bounded staged and unstaged repository diffs through git_diff.","tags":["compat","repo","git","diff"],
        "parameters":{"max_chars":P(20000,"Maximum characters per diff")},
        "pipeline":[{"id":"result","tool":"git_diff","args":{"max_chars":{"$param":"max_chars","default":20000}}}],
    },
    {
        "key":"compat.diff_observations","version":1,"name":"compat.diff_observations","target_tool":"diff_observations",
        "description":"Load two durable observations and compare their content through text_diff.","tags":["compat","observation","diff","monitor"],
        "parameters":{"old_id":P("","Older observation id"),"new_id":P("","Newer observation id"),"max_diff_chars":P(12000,"Maximum diff characters")},
        "pipeline":[
            {"id":"old","tool":"observation_get","args":{"observation_id":{"$param":"old_id"}}},
            {"id":"new","tool":"observation_get","args":{"observation_id":{"$param":"new_id"}}},
            {"id":"diff","tool":"text_diff","args":{"a":{"$ref":"old","path":"content"},"b":{"$ref":"new","path":"content"},"max_chars":{"$param":"max_diff_chars","default":12000}}},
            {"id":"result","tool":"compose_object","args":{"data":{"old_id":{"$param":"old_id"},"new_id":{"$param":"new_id"},"diff":{"$ref":"diff"}}}},
        ],
    },
]

NATIVE_ONLY = {
    "get_repo_map": "AST/repository indexing is a specialized index builder rather than a monolithic composition target",
    "search_repo_symbols": "query over the repository symbol index is already a narrow lookup",
    "read_repo_symbol": "bounded source/symbol retrieval is already a narrow lookup",
    "repo_checks": "runs isolated compile/config/lint/test commands against a temporary repository copy and is intentionally native",
    "dependency_audit": "combines Python distribution metadata and many external command checks; keeping one native audit avoids a very large fixed recipe",
    "tool_health": "introspects the live registry and dependency metadata, which is harness control state rather than ordinary primitive data",
}
