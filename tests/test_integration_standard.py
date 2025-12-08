import unittest
from unittest.mock import MagicMock, AsyncMock
from agentmesh.builder import AgentBuilder
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.runtimes.base_client import ModelClient

class MockLLM(ModelClient):
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        # Return next response
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return "Generic response"
        
    async def agenerate(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

class TestIntegrationStandard(unittest.IsolatedAsyncioTestCase):
    
    async def test_standard_flow(self):
        # 1. Setup Mocks
        # Sequence of LLM calls:
        # 1. Planner: Plan a search
        # 2. Router (if LLM used, but let's assume Router uses MCP match or static for this test to keep simple, 
        #    actually Router default behavior might check LLM if no direct match. 
        #    Let's ensure Planner output matches a tool so Router uses MCP match.)
        # 3. Validator: Done
        # 4. Composer: Final Answer
        
        # Planner response
        plan_resp = '[{"task": "mock_tool", "args": {"input": "test"}}]'
        # Validator response
        val_resp = '{"done": true, "reason": "satisfied"}'
        # Composer response
        comp_resp = "The answer is found."
        
        llm = MockLLM([plan_resp, val_resp, comp_resp])
        
        # Mock Manager
        mgr = MagicMock(spec=MCPManager)
        mgr.list_tools.return_value = ["mock_tool"]
        mgr.invoke_tool = AsyncMock(return_value="tool_output_data")
        # Router needs to see tool list too
        mgr.get_tool.return_value = (MagicMock(), None) 
        
        # 2. Build Graph
        builder = AgentBuilder()
        builder.mcp_manager = mgr
        builder.build_nodes(planner_llm=llm, composer_llm=llm, validator_llm=llm)
        
        # Manually inject manager into router cache if needed, 
        # but builder.build_nodes passes manager to router, 
        # and router calls get_tool_list which uses manager.list_tools
        
        graph = builder.compile_graph()
        
        # 3. Run
        initial_state = {
            "user_query": "find something",
            "results": {},
            "loops": 0,
            "max_loops": 3
        }
        
        final_state = await graph.ainvoke(initial_state)
        
        # 4. Assertions
        # Check final answer
        self.assertEqual(final_state["final_answer"], "The answer is found.")
        
        # Check tool execution results
        results = final_state["results"]
        self.assertIn("mock_tool", results)
        self.assertEqual(results["mock_tool"], "tool_output_data")
        
        # Verify call counts?
        # Planner called (1) -> Validator called (1) -> Composer called (1)
        # Total 3 LLM calls
        self.assertEqual(llm.call_count, 3)
        
        # Verify tool invocation
        mgr.invoke_tool.assert_called_with("mock_tool", input="test")

