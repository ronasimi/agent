import json
import subprocess
from pathlib import Path


def test_recipe_decision_state_persists_and_is_conversation_scoped():
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "interaction_state.js"
    script = f"""
const {{RecipeDecisionStore}} = require({json.dumps(str(module))});
const values = new Map();
const storage = {{
  getItem: key => values.has(key) ? values.get(key) : null,
  setItem: (key, value) => values.set(key, value),
}};
let store = new RecipeDecisionStore({{storage, key:'test', limit:20}});
store.upsert({{id:'c1:r1', conversationId:'c1', message:'Save it?', createdAt:'2026-09-21T00:00:00Z'}});
store.upsert({{id:'c2:r2', conversationId:'c2', message:'Save another?', createdAt:'2026-09-21T00:01:00Z'}});
store.decide('c1:r1', 'up');
store.decide('c2:r2', 'down');
store = new RecipeDecisionStore({{storage, key:'test', limit:20}});
const before = {{c1:store.list('c1'), c2:store.list('c2')}};
store.removeConversation('c1');
console.log(JSON.stringify({{before, after:store.list('c1')}}));
"""
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    assert data["before"]["c1"][0]["decision"] == "up"
    assert data["before"]["c2"][0]["decision"] == "down"
    assert data["after"] == []


def test_recipe_decision_store_recovers_from_corrupt_storage():
    root = Path(__file__).resolve().parents[1]
    module = root / "webui" / "static" / "interaction_state.js"
    script = f"""
const {{RecipeDecisionStore}} = require({json.dumps(str(module))});
const storage = {{getItem: () => '{{broken', setItem: () => {{}}}};
const store = new RecipeDecisionStore({{storage, key:'test'}});
console.log(JSON.stringify(store.list('anything')));
"""
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []
