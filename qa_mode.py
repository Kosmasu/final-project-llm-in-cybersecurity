from typing import Literal
from pydantic import BaseModel

from conversations import Conversation
from llm import LLM
from search import SearchResult
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
@observe()
def determine_mode(llm: LLM, user_query: str) -> Mode | None:
    convo: Conversation = Conversation.from_system_message("""
You are an AI assistant that categorizes user queries into one of two modes: 'phishing_detection' or 'qa'.
- 'phishing_detection' mode: Use this mode when the user provides a piece of text (like an email, URL, or message) and asks you to analyze *that specific text* to determine if it is a phishing attempt or safe. The user wants a verdict on the provided content.
- 'qa' mode: Use this mode when the user is asking a general question, seeking information, definitions, explanations, or advice. This includes questions *about* phishing, cybersecurity concepts, or any other topic. The user is looking for knowledge, not a verdict on a specific piece of text they've provided for analysis.

Carefully analyze the user's query to understand their primary intent.
    """.strip()).add_user_message(f"""
# Instructions
Based on the user query below, determine if the primary intent is to have a specific piece of text analyzed for phishing ('phishing_detection') or to ask a question/seek information ('qa').
Provide your reasoning.

# Examples of Classification

## Example 1
User Query: "Is this email a scam? 'Subject: Urgent Account Verification. Dear user, click here http://example.com/login to verify your account.'"
Output:
{{
    "reason": "The user provided an email text and asked if it's a scam, indicating a request for phishing analysis on the provided text.",
    "mode": "phishing_detection"
}}

## Example 2
User Query: "What are the common signs of a phishing email?"
Output:
{{
    "reason": "The user is asking for general information about phishing signs, not asking to analyze a specific email.",
    "mode": "qa"
}}

## Example 3
User Query: "Can you check if 'http://suspicious-link.com/update-your-details' is safe?"
Output:
{{
    "reason": "The user provided a URL and asked if it's safe, requesting an analysis of that specific URL.",
    "mode": "phishing_detection"
}}

## Example 4
User Query: "Tell me about spear phishing."
Output:
{{
    "reason": "The user is asking for an explanation of a cybersecurity term.",
    "mode": "qa"
}}

## Example 5
User Query: "what is a phishing attack?"
Output:
{{
    "reason": "The user is asking for a definition of 'phishing attack', which is a request for information.",
    "mode": "qa"
}}

## Example 6
User Query: "
```
Subject: Important Update Required
Dear User,
We have detected unusual activity in your account. Please click the link below to verify your identity:
http://malicious-link.com/verify
Failure to do so may result in account suspension.
Best regards,
Your Bank
```
Is this email safe?"
Output:
{{
    "reason": "The user provided an email text and asked if it's safe, indicating a request for phishing analysis on the provided text.",
    "mode": "phishing_detection"
}}

# User Query to Analyze
{user_query}

# Required Output Format
Return a JSON object with "reason" and "mode".
{{
    "reason": "Your reasoning here.",
    "mode": "phishing_detection" | "qa"
}}
    """.strip())
    return utils.extract_and_validate_json(Mode, llm.generate(convo))


@observe()
def answer_question(
    llm: LLM, convo: Conversation, user_query: str, context: list[SearchResult] | None = None
) -> str:
    system_message = "You are a question-answering assistant that is well-versed in cybersecurity. Your task is to answer the user's query."

    context_prompt_addition = ""
    if context:
        context_str = "\n\n".join([f"## Search Result {index}\n{doc.format()}" for index, doc in enumerate(context, start=1)])
        context_prompt_addition = f"""
# Context
Please use the following context to answer the user's query. If the context is not relevant, answer based on your general knowledge.
{context_str}
        """.strip()

    _convo: Conversation = Conversation.from_system_message(
        f"{system_message}\n{context_prompt_addition}"
    )
    _convo.messages += convo.messages
    _convo.add_user_message(user_query)
    return llm.generate(_convo)
