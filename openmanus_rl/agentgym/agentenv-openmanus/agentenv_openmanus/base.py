from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """Minimal base class for tools used in the OpenManus environment."""

    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Return the OpenAI function-call representation for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self) -> bool:  # pragma: no cover - simple convenience method
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult") -> "ToolResult":
        def combine(field: Optional[str], other_field: Optional[str], cat: bool = True):
            if field and other_field:
                if cat:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine(self.output, other.output),
            error=combine(self.error, other.error),
            base64_image=combine(self.base64_image, other.base64_image, False),
            system=combine(self.system, other.system),
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Error: {self.error}" if self.error else self.output

    def replace(self, **kwargs) -> "ToolResult":  # pragma: no cover - trivial
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
