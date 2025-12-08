from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class PlannerContext:
    """
    Structured context for the Planner node.
    
    Attributes:
        history: List of conversation messages.
        memory: Long-term or session-based memory state.
    """
    history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PlannerContext":
        """
        Factory method to create PlannerContext from a raw state dictionary.
        """
        # Attempt to extract history and memory using common keys, 
        # but defaulting to empty if not found.
        # This allows for partial adoption.
        return cls(
            history=state.get("history", []) or state.get("messages", []),
            memory=state.get("memory", {})
        )

    def format_for_prompt(self) -> str:
        """
        Formats the context into a string suitable for injection into an LLM prompt.
        """
        lines = []
        
        if self.memory:
            lines.append("Memory/State:")
            lines.append(json.dumps(self.memory, indent=2))
            lines.append("")
            
        if self.history:
            lines.append("Conversation History:")
            # Simple formatting - can be improved to limit length
            for msg in self.history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate content for display if too long? For now keep it simple.
                lines.append(f"- {role}: {content}")
                
        if not lines:
            return "No context available."
            
        return "\n".join(lines)
