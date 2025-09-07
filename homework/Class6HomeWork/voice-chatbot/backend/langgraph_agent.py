from typing import Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from sympy import sympify
import json
from services.arxiv_service import search_arxiv as arxiv_search_service


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for academic papers related to the query."""
    return arxiv_search_service(query)


@tool 
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error with calculation: {e}"


class LangGraphAgent:
    def __init__(self, model_inference_func):
        self.model_inference_func = model_inference_func
        self.tools = [search_arxiv, calculate]
        self.tool_node = ToolNode(self.tools)
        self.memory = MemorySaver()
        self.graph = self._create_graph()
        
    def _create_graph(self):
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _call_model(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage):
            user_text = last_message.content
        else:
            user_text = str(last_message.content)
            
        # Use the existing model inference
        response = self.model_inference_func(user_text)
        
        # Try to parse as tool call first
        try:
            parsed = json.loads(response)
            if "function" in parsed and "arguments" in parsed:
                func_name = parsed["function"]
                args = parsed.get("arguments", {})
                
                # Convert to tool call format
                if func_name == "search_arxiv":
                    tool_call = {
                        "name": "search_arxiv",
                        "args": {"query": args.get("query", "")},
                        "id": "call_1"
                    }
                elif func_name == "calculate":
                    tool_call = {
                        "name": "calculate", 
                        "args": {"expression": args.get("expression", "")},
                        "id": "call_1"
                    }
                else:
                    # Unknown function, return as regular message
                    return {"messages": [AIMessage(content=response)]}
                
                return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Regular response
        return {"messages": [AIMessage(content=response)]}
    
    def _should_continue(self, state: MessagesState) -> Literal["continue", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"
    
    def invoke(self, user_text: str, thread_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config
        )
        
        # Return the last message content (could be AI or Tool message)
        for message in reversed(result["messages"]):
            if hasattr(message, 'content') and message.content:
                return message.content
        
        return "No response generated"


def create_langgraph_agent(model_inference_func):
    """Factory function to create a LangGraph agent"""
    return LangGraphAgent(model_inference_func)