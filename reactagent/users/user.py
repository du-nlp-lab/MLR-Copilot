from abc import ABC, abstractmethod

class User(ABC):
    """Base class for interface to user"""
    @abstractmethod
    def interact(relevant_history: str, entries: dict, observation: str) -> str:
        pass


    @staticmethod
    def indent_text(s, n):
        return "\n".join(" "*n + line for line in s.split("\n"))

