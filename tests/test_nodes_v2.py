import unittest
from unittest.mock import MagicMock, AsyncMock
import json

from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer
from agentmesh.mcp.mcp_manager import MCPManager

# Mock classes
class MockLLM:
    def __init__(self, response):
        self.response = response
        
    def generate(self, prompt, **kwargs):
        return self.response
        
    async def agenerate(self, prompt, **kwargs):
        return self.response

class TestNodesV2(unittest.IsolatedAsyncioTestCase):
    
    async def test_router_llm(self):
        # Mocks
        resp = '{"tool": "search", "args": {"q": "python"}}'
        llm = MockLLM(resp)
        mgr = MagicMock(spec=MCPManager)
        mgr.list_tools.return_value = ["search", "calc"]
        
        router = Router(model_client=llm, mcp_manager=mgr)
        
        # Test routing via LLM (via run)
        state = {"todos": [{"task": "find info", "args": {"q": "python"}}], "results": {}}
        output = await router.run(state)
        
        routed_items = output["todos"]
        self.assertEqual(len(routed_items), 1)
        route = routed_items[0]["route"]
        self.assertEqual(route["tool"], "search")
        self.assertEqual(route["args"]["q"], "python")
        self.assertEqual(output["tactical_status"], "CONTINUE")

    async def test_router_mcp_exact_match(self):
        mgr = MagicMock(spec=MCPManager)
        mgr.list_tools.return_value = ["filesystem.read_file"]
        router = Router(mcp_manager=mgr)
        
        # Exact match
        todo = {"task": "filesystem.read_file", "args": {"path": "/tmp/test"}}
        state = {"todos": [todo], "results": {}}
        
        output = await router.run(state)
        route = output["todos"][0]["route"]
        self.assertEqual(route["tool"], "filesystem.read_file")
        
        # Tail match
        todo2 = {"task": "read_file", "args": {"path": "/tmp/test"}}
        state2 = {"todos": [todo2], "results": {}}
        output2 = await router.run(state2)
        route2 = output2["todos"][0]["route"]
        self.assertEqual(route2["tool"], "filesystem.read_file")

    async def test_executor_async(self):
        mgr = MagicMock(spec=MCPManager)
        # Setup async mock for invoke_tool
        mgr.invoke_tool = AsyncMock(return_value="file content")
        
        executor = Executor(mcp_manager=mgr)
        route = {"tool": "read_file", "args": {"path": "foo"}}
        
        # Test granular execute
        res = await executor.execute(route)
        self.assertEqual(res, "file content")
        mgr.invoke_tool.assert_called_with("read_file", path="foo")

    async def test_validator_success(self):
        llm = MockLLM('{"done": true, "reason": "found it"}')
        validator = Validator(model_client=llm)
        state = {"user_query": "foo", "results": {"t1": "bar"}}
        
        output = await validator.run(state)
        self.assertTrue(output["done"])
        self.assertEqual(output["validator_info"]["reason"], "found it")

    async def test_composer(self):
        llm = MockLLM("The answer is 42.")
        composer = Composer(llm=llm)
        state = {"user_query": "what is 6*7", "results": {"calc": 42}}
        
        output = await composer.run(state)
        self.assertEqual(output["final_answer"], "The answer is 42.")
