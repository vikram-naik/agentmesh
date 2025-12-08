import unittest
from unittest.mock import MagicMock
from agentmesh.nodes.planner import Planner
from agentmesh.nodes.context import PlannerContext

# Mock LLM Client
class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text
    
    def generate(self, prompt, max_tokens=1000):
        return self.response_text

class TestPlanner(unittest.TestCase):

    def test_parse_json_simple(self):
        llm = MockLLM('[{"task": "search", "args": {"q": "foo"}}]')
        planner = Planner(llm=llm)
        todos = planner._parse_json_response(llm.response_text)
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["task"], "search")

    def test_parse_json_markdown(self):
        text = """Here is the plan:
```json
[
    {"task": "think", "args": {}}
]
```
"""
        llm = MockLLM(text)
        planner = Planner(llm=llm)
        todos = planner._parse_json_response(text)
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["task"], "think")

    def test_parse_failure_recovery(self):
        text = "I cannot do that."
        planner = Planner(llm=MockLLM(text))
        todos = planner._parse_json_response(text)
        self.assertEqual(todos[0]["task"], "unknown")

class TestPlannerAsync(unittest.IsolatedAsyncioTestCase):
    async def test_planner_async_flow(self):
        # Setup
        llm = MockLLM('[{"task": "test", "args": {}}]')
        planner = Planner(llm=llm)
        state = {"user_query": "hello", "history": []}
        
        # Act
        plan = await planner.plan(state)
        
        # Assert
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["task"], "test")
