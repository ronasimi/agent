from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path
from .text import text_diff

def json_query(path_expr: str = "$", data: Any = "", path: str = "") -> str:
    """Select a simple dotted path from JSON text or a workspace JSON file."""
    try: return _json(_get_path(_load_json(data,path),path_expr))
    except Exception as exc: return f"Error: json_query failed: {exc}"

def json_filter(key: str, operator: str, value: str, data: Any = "", path: str = "", limit: int = 200) -> str:
    """Filter a JSON array of objects using one simple key comparison."""
    try:
        items=_load_json(data,path)
        if not isinstance(items,list): return "Error: JSON root must be an array."
        try: wanted=json.loads(value)
        except Exception: wanted=value
        ops={"eq":lambda a,b:a==b,"ne":lambda a,b:a!=b,"gt":lambda a,b:a>b,"ge":lambda a,b:a>=b,"lt":lambda a,b:a<b,"le":lambda a,b:a<=b,"contains":lambda a,b:str(b) in str(a)}
        if operator not in ops:return "Error: unsupported operator."
        out=[]
        for item in items:
            if not isinstance(item,dict): continue
            try:
                if ops[operator](_get_path(item,key),wanted): out.append(item)
            except Exception: pass
            if len(out)>=_bounded_int(limit,1,500): break
        return _json(out)
    except Exception as exc:return f"Error: json_filter failed: {exc}"

def json_sort(key: str, data: Any = "", path: str = "", descending: bool = False, limit: int = 200) -> str:
    """Sort a JSON array of objects by a dotted key."""
    try:
        items=_load_json(data,path)
        if not isinstance(items,list): return "Error: JSON root must be an array."
        items=sorted(items,key=lambda x:(_get_path(x,key) is None,_get_path(x,key)),reverse=bool(descending))
        return _json(items[:_bounded_int(limit,1,500)])
    except Exception as exc:return f"Error: json_sort failed: {exc}"

def json_diff(a: Any, b: Any) -> str:
    """Compare two JSON values and return added/removed/changed paths."""
    try:
        left=_load_json(a); right=_load_json(b); changes=[]
        def walk(x,y,p="$",depth=0):
            if len(changes)>=500 or depth>20:return
            if type(x)!=type(y): changes.append({"path":p,"old":x,"new":y});return
            if isinstance(x,dict):
                for k in sorted(set(x)|set(y)):
                    q=f"{p}.{k}"
                    if k not in x:changes.append({"path":q,"added":y[k]})
                    elif k not in y:changes.append({"path":q,"removed":x[k]})
                    else:walk(x[k],y[k],q,depth+1)
            elif isinstance(x,list):
                if x!=y:changes.append({"path":p,"old":x[:50],"new":y[:50]})
            elif x!=y:changes.append({"path":p,"old":x,"new":y})
        walk(left,right)
        return _json({"changed":bool(changes),"changes":changes})
    except Exception as exc:return f"Error: json_diff failed: {exc}"

def json_head(data: Any = "", path: str = "", count: int = 10) -> str:
    """Return the first N items from a JSON array."""
    try:
        items=_load_json(data,path)
        if not isinstance(items,list):return "Error: JSON root must be an array."
        return _json(items[:_bounded_int(count,1,500)])
    except Exception as exc:return f"Error: json_head failed: {exc}"

def json_count(data: Any = "", path: str = "") -> str:
    """Return the length of a JSON array/object/string."""
    try:
        value=_load_json(data,path); return _json({"count":len(value) if hasattr(value,"__len__") else 1,"type":type(value).__name__})
    except Exception as exc:return f"Error: json_count failed: {exc}"

def yaml_query(path_expr: str = "$", text: str = "", path: str = "") -> str:
    """Select a simple dotted path from YAML text or a workspace YAML file."""
    try:
        import yaml
        raw=_source_text(text,path); return _json(_get_path(yaml.safe_load(raw),path_expr))
    except Exception as exc:return f"Error: yaml_query failed: {exc}"

def csv_query(column: str = "", equals: str = "", text: str = "", path: str = "", limit: int = 100) -> str:
    """Read bounded CSV rows and optionally filter one column by exact string value."""
    import io
    try:
        raw=_source_text(text,path); reader=csv.DictReader(io.StringIO(raw)); out=[]
        for row in reader:
            if column and str(row.get(column,"")) != str(equals):continue
            out.append(dict(row))
            if len(out)>=_bounded_int(limit,1,500):break
        return _json(out)
    except Exception as exc:return f"Error: csv_query failed: {exc}"

def compare_json(a: Any, b: Any) -> str:
    """Alias for structured JSON comparison."""
    return json_diff(a,b)

def compare_values(a: Any, b: Any) -> str:
    """Compare two scalar/string values and report equality plus representations."""
    return _json({"equal":a==b,"a":a,"b":b})

def compare_text(a: str, b: str) -> str:
    """Alias for unified text comparison."""
    return text_diff(a,b)

def compose_object(data: dict) -> str:
    """Return an object unchanged as normalized JSON; useful as the final aggregation stage in a pipeline."""
    return _json(data)


def compose_list(items: list) -> str:
    """Return a list unchanged as normalized JSON; useful as the final aggregation stage in a pipeline."""
    return _json(items)


def choose_value(condition: bool, if_true: Any = None, if_false: Any = None) -> str:
    """Return one of two JSON-compatible values based on a boolean condition."""
    return _json(if_true if bool(condition) else if_false)


def map_value(value: Any, mapping: dict, default: Any = None) -> str:
    """Map a scalar value through a bounded explicit mapping and return the mapped value."""
    key = str(value)
    if key in mapping:
        return _json(mapping[key])
    return _json(default if default is not None else value)
