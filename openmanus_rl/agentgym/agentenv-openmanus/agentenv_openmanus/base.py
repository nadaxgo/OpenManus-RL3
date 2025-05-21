from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """Minimal tool interface for the OpenManus environment."""

    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    def to_param(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self) -> bool:
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine(field: Optional[str], other_field: Optional[str], concat: bool = True):
            if field and other_field:
                if concat:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine(self.output, other.output),
            error=combine(self.error, other.error),
            base64_image=combine(self.base64_image, other.base64_image, False),
            system=combine(self.system, other.system),
        )

    def __str__(self) -> str:
        return f"Error: {self.error}" if self.error else str(self.output)

    def replace(self, **kwargs) -> "ToolResult":
        return type(self)(**{**self.dict(), **kwargs})
