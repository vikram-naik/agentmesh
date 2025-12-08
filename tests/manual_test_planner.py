
import json
from agentmesh.nodes.planner import Planner

class MockLLM:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt, max_tokens=300):
        self.last_prompt = prompt
        return json.dumps([{"task": "test_ok", "args": {}}])

def test_planner_feedback():
    llm = MockLLM()
    planner = Planner(llm)

    # State with validator feedback
    state = {
        "user_query": "Find fiscal policy details",
        "results": {
            "_validator": {
                "reason": "Missing details for 2024.",
                "todo_hints": [{"task": "search", "args": {"query": "fiscal policy 2024"}}]
            }
        }
    }

    print("Running plan()...")
    planner.plan(state)

    print("\n--- Captured Prompt ---")
    print(llm.last_prompt)
    print("-----------------------")

    # Assertions
    if "Feedback from previous attempt:" in llm.last_prompt:
        print("[PASS] Feedback section found.")
    else:
        print("[FAIL] Feedback section missing.")

    if "Missing details for 2024" in llm.last_prompt:
         print("[PASS] Specific reason found.")
    else:
         print("[FAIL] Specific reason missing.")

if __name__ == "__main__":
    test_planner_feedback()
