from typing import Generator
from pydantic import BaseModel
import streamlit as st

from conversations import Conversation
from embedding import create_embedding_model, load_bm25_retriever
from llm import LLM, LLMName
from phishing_mode import classify_phishing_pretrained
from search import search_and_fetch_contents  # Added import


from qa_mode import Mode, answer_question, determine_mode
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, RETRIEVER_PATH

create_embedding_model()

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host="https://cloud.langfuse.com",
)

ID_LLM_BASE = "llm_base"
ID_LLM_QA = "llm_qa"
ID_LLM_PHISHING = "llm_phishing"
ID_CONVO = "convo"
ID_RETRIEVER = "retriever"

if ID_LLM_BASE not in st.session_state:
    llm_base: LLM = LLM(model_name=LLMName.LLAMA_3_1_8B)
    st.session_state[ID_LLM_BASE] = llm_base
else:
    llm_base: LLM = st.session_state[ID_LLM_BASE]

if ID_LLM_QA not in st.session_state:
    llm_qa: LLM = LLM(model_name=LLMName.LLAMA_3_1_CHATQA)
    st.session_state[ID_LLM_QA] = llm_qa
else:
    llm_qa: LLM = st.session_state[ID_LLM_QA]

if ID_LLM_PHISHING not in st.session_state:
    llm_phishing: LLM = LLM(model_name=LLMName.GEMMA_1B_FINETUNED)
    st.session_state[ID_LLM_PHISHING] = llm_phishing
else:
    llm_phishing: LLM = st.session_state[ID_LLM_PHISHING]

if ID_CONVO not in st.session_state:
    convo: Conversation = Conversation()
    st.session_state[ID_CONVO] = convo
else:
    convo: Conversation = st.session_state[ID_CONVO]

if ID_RETRIEVER not in st.session_state:
    loaded_bm25_retriever = load_bm25_retriever(RETRIEVER_PATH)
    st.session_state[ID_RETRIEVER] = loaded_bm25_retriever
else:
    loaded_bm25_retriever = st.session_state[ID_RETRIEVER]


def reset():
    convo.messages = []
    st.session_state[ID_CONVO] = convo


st.title("Final Project")

st.button("Reset", on_click=reset)

for message in convo.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)


class Alert(BaseModel):
    message: str


def ask_question(user_message: str) -> Generator[Alert | str, None, None]:
    # Put main into here
    mode = determine_mode(llm=llm_base, user_query=user_message)
    if mode is None:
        mode = Mode(
            reason="Failed to determine mode. Defaulting to QA.",
            mode="qa",
        )

    trace_id = langfuse_context.get_current_trace_id()
    if trace_id:
        langfuse.event(
            trace_id=trace_id,
            name=f"{mode.mode} mode",
            input=user_message,
            output=mode,
        )

    if mode.mode == "qa":
        # QA mode pipeline here
        yield Alert(
            message="Q&A mode detected. Performing web search for relevant information..."
        )

        # Perform web search to get context
        context_docs = search_and_fetch_contents(user_message, num_results=3)

        if context_docs:
            yield Alert(
                message=f"Found {len(context_docs)} relevant document(s). Generating response..."
            )
        else:
            yield Alert(
                message="No specific documents found via web search. Generating response based on general knowledge..."
            )

        response = answer_question(
            llm=llm_qa,
            convo=convo,
            user_query=user_message,
            context=context_docs,  # Pass fetched context
        )
    else:
        # Phishing detection mode pipeline here
        yield Alert(message="Phishing detection mode detected. Classifying email...")

        phishing_evaluation = classify_phishing_pretrained(
            llm=llm_phishing,
            retriever=loaded_bm25_retriever,
            user_query=user_message,
        )
        if not phishing_evaluation:
            response = "Failed to classify email."
            yield Alert(message=response)
        else:
            yield Alert(message="Phishing evaluation completed. Generating response...")
            response = f"""
### Phishing Evaluation
Email is detected as {"**PHISHING**" if phishing_evaluation.is_phishing else "**NOT PHISHING**"}.
### Reason
{phishing_evaluation.reason}
### Explanation
{phishing_evaluation.explanation}
""".strip()

    convo.add_user_message(user_message)
    convo.add_assistant_message(response)
    langfuse_context.update_current_observation(
        output=response,
    )

    st.session_state[ID_CONVO] = convo
    yield response


@observe(capture_input=False, capture_output=False, name="answer")
def main():
    if user_message := st.chat_input("Type a message..."):
        with st.chat_message("user"):
            st.markdown(user_message)
        langfuse_context.update_current_observation(
            input=user_message,
        )

        for output in ask_question(user_message):
            if isinstance(output, str):
                with st.chat_message("assistant"):
                    st.markdown(output)
            elif isinstance(output, Alert):
                alert = output
                with st.chat_message("alert", avatar=":material/settings:"):
                    st.markdown(alert.message)


main()
