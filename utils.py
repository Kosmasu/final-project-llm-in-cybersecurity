from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
import json

T = TypeVar("T", bound=BaseModel)

def extract_json_string(result: str) -> str | None:
    start = result.find("{")
    if start == -1:
        return None

    stack = []
    in_string = False
    escape = False
    for i, c in enumerate(result[start:]):
        if c == '"' and not escape:
            in_string = not in_string
        elif c == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if c == '{':
                stack.append('{')
            elif c == '}':
                stack.pop()
                if not stack:
                    return result[start:start + i + 1]
        escape = False
    return None

def extract_and_validate_json(
    cls: Type[T],
    input: str,
) -> T | None:
    string = extract_json_string(input)
    if not string:
        return None
    try:
        return cls.model_validate_json(string)
    except ValidationError:
        return None

def sanitize_email(email: str) -> str:
    """
    Sanitize the email by removing any unwanted characters or formatting.
    """
    # Remove any unwanted characters or formatting
    sanitized_email = email.replace("\n", " ").replace("\r", " ")
    return sanitized_email