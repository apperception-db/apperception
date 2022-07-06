from typing import Any


class View:
    def __init__(self, view_name: str):
        self.view_name: str = view_name
        self.default: bool = False

    def from_context(self, context: Any):
        self.context = context

    def resolve_key(self, column_key: str):
        if column_key in self.__class__.__dict__:
            return self.__class__.__dict__[column_key]
        else:
            return None

    def contain(self, column_key: str):
        return column_key in self.__dict__.keys()
