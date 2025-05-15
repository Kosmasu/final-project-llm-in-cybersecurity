from pydantic import BaseModel
from llama_index.core.base.base_retriever import BaseRetriever

from conversations import Conversation
from embedding import find_similar_emails
from llm import LLM
import utils
from langfuse.decorators import observe


class PhishingEvaluation(BaseModel):
    reason: str
    explanation: str
    is_phishing: bool


# Classify the email as phishing or not using pretrained model
@observe()
def classify_phishing_pretrained(
    llm: LLM, retriever: BaseRetriever, user_query: str
) -> PhishingEvaluation | None:
    # Fetch similar emails from the database using #embedding.py
    emails = find_similar_emails(retriever, user_query)
    emails_text = "\n".join(
        [
            f"### Email {idx}\n{email.format()}"
            for idx, email in enumerate(emails, start=1)
        ]
    )

    convo: Conversation = Conversation.from_system_message("""
        You are a phishing detection assistant. Your task is to determine if the provided email is a phishing attempt or not.
        If it is a phishing attempt, respond with "true". If it is not, respond with "false".
    """).add_user_message(f"""
        # User Query
        {user_query}

        # Similar Emails
        {emails_text}

        # Instructions
        Determine if the provided email is a phishing attempt or not. If it is, respond with "true". If it is not, respond with "false".
        Explain the reasoning behind the phishing classification of the provided email. Explain it to the user in a way that they can understand. Use simple language and avoid technical jargon.
        Follow the format below.

        # Format
        {{
            "reason": "The reason for the classification",
            "explanation": "The explanation of the classification to the user",
            "is_phishing": true | false
        }}
    """)
    return utils.extract_and_validate_json(PhishingEvaluation, llm.generate(convo))
