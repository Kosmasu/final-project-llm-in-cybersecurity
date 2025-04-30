from pydantic import BaseModel

from conversations import Conversation
from llm import LLM
import utils
from langfuse.decorators import observe

class PhishingEvaluation(BaseModel):
    reason: str
    is_phishing: bool

# Classify the email as phishing or not using pretrained model
@observe()
def classify_phishing_pretrained(llm: LLM, email: str) -> PhishingEvaluation | None:
    convo: Conversation = Conversation.from_system_message("""
        You are a phishing detection assistant. Your task is to determine if the provided email is a phishing attempt or not.
        If it is a phishing attempt, respond with "true". If it is not, respond with "false".
    """).add_user_message(f"""
        # Email
        ```
        {email}
        ```

        # Instructions
        Determine if the provided email is a phishing attempt or not. If it is, respond with "true". If it is not, respond with "false".
        Follow the format below.

        # Format
        {{
            "reason": "The reason for the classification",
            "is_phishing": true | false
        }}
    """)
    return utils.extract_and_validate_json(PhishingEvaluation, llm.generate(convo))

@observe()
def explain_phishing_evaluation(llm: LLM, email: str, phishing_evaluation: PhishingEvaluation) -> str:
    convo: Conversation = Conversation.from_system_message("""
        You are a phishing detection assistant. Your task is to explain the reasoning behind the phishing classification of the provided email.
    """).add_user_message(f"""
        # Email
        ```
        {email}
        ```

        # Phishing Evaluation
        This email is classified as `{phishing_evaluation.is_phishing}` because {phishing_evaluation.reason}.

        # Instructions
        Explain the reasoning behind the phishing classification of the provided email. Explain it to the user in a way that they can understand. Use simple language and avoid technical jargon.
    """)
    return llm.generate(convo)
