from typing import Literal
from pydantic import BaseModel

from conversations import Conversation
from llm import LLM
import utils
from langfuse.decorators import observe


# Mode Explanation
# QA: This is a question-answering mode where the model provides answers to user queries.
# Phishing Detection: This mode is specifically designed to identify phishing attempts in the provided text.
class Mode(BaseModel):
    reason: str
    mode: Literal["qa", "phishing_detection"]


# Check user's query to determine the mode
# This function checks the user's query to determine if it is a question-answering or phishing detection request.
@observe()
def determine_mode(llm: LLM, user_query: str) -> Mode | None:
    convo: Conversation = Conversation.from_system_message("""
        You are a phishing detection assistant. Your task is to determine if the user's query is related to phishing detection or not.
        If the query is related to phishing detection, respond with "phishing_detection". If it is not, respond with "qa".
    """).add_user_message(f"""
        # User Query
        {user_query}

        # Instructions
        Determine if the user's query is related to phishing detection or not. If it is, respond with "phishing_detection". If it is not, respond with "qa".
        Follow the format below.

        # Format
        {{
            "reason": "The reason for the classification",
            "mode": "phishing_detection" | "qa"
        }}
    """)
    return utils.extract_and_validate_json(Mode, llm.generate(convo))

@observe()
def answer_question(llm: LLM, convo: Conversation, user_query: str) -> str:
    _convo: Conversation = Conversation.from_system_message("""
        You are a question-answering assistant that is well-versed in cybersecurity. Your task is to answer the user's query.
    """)
    _convo.messages += convo.messages
    _convo.add_user_message(user_query)
    return llm.generate(_convo)