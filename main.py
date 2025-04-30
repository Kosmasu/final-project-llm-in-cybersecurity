import streamlit as st

from conversations import Conversation
from llm import LLM, LLMName
from phishing_detection import (
    classify_phishing_pretrained,
    explain_phishing_evaluation,
)


from qa import Mode, answer_question, determine_mode
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe


langfuse = Langfuse()

st.title("Final Project")

ID_LLM_BASE = "llm_base"
ID_LLM_QA = "llm_qa"
ID_LLM_PHISHING = "llm_phishing"
ID_CONVO = "convo"

if ID_LLM_BASE not in st.session_state:
    llm_base: LLM = LLM(model_name=LLMName.LLAMA_3_1_8B)
    st.session_state[ID_LLM_BASE] = llm_base
else:
    llm_base: LLM = st.session_state[ID_LLM_BASE]

if ID_LLM_QA not in st.session_state:
    llm_qa: LLM = LLM(model_name=LLMName.LLAMA_3_1_8B)
    st.session_state[ID_LLM_QA] = llm_qa
else:
    llm_qa: LLM = st.session_state[ID_LLM_QA]

if ID_LLM_PHISHING not in st.session_state:
    llm_phishing: LLM = LLM(model_name=LLMName.LLAMA_3_1_8B)
    st.session_state[ID_LLM_PHISHING] = llm_phishing
else:
    llm_phishing: LLM = st.session_state[ID_LLM_PHISHING]

if ID_CONVO not in st.session_state:
    convo: Conversation = Conversation()
    st.session_state[ID_CONVO] = convo
else:
    convo: Conversation = st.session_state[ID_CONVO]

for message in convo.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)


@observe(capture_input=False, capture_output=False, name="answer")
def main():
    if user_message := st.chat_input("Type a message..."):
        with st.chat_message("user"):
            st.markdown(user_message)
        langfuse_context.update_current_observation(
            input=user_message,
        )

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
            # TODO: Implement the QA mode pipeline: RAG, etc.
            with st.chat_message("alert", avatar=":material/settings:"):
                st.markdown("Q&A mode detected. Generating response...")

            response = answer_question(
                llm=llm_qa,
                convo=convo,
                user_query=user_message,
            )
        else:
            # Phishing detection mode pipeline here
            # TODO: Change to finetuned model, and RAG on similar emails
            with st.chat_message("alert", avatar=":material/settings:"):
                st.markdown("Phishing detection mode detected. Classifying email...")

            phishing_evaluation = classify_phishing_pretrained(
                llm=llm_phishing,
                email=user_message,
            )
            if not phishing_evaluation:
                response = "Failed to classify email."
                with st.chat_message("alert", avatar=":material/settings:"):
                    st.markdown(response)
            else:
                with st.chat_message("alert", avatar=":material/settings:"):
                    st.markdown("Phishing evaluation completed. Generating response...")
                response = explain_phishing_evaluation(
                    llm=llm_qa,
                    email=user_message,
                    phishing_evaluation=phishing_evaluation,
                )

        with st.chat_message("assistant"):
            st.markdown(response)
        convo.add_user_message(user_message)
        convo.add_assistant_message(response)
        langfuse_context.update_current_observation(
            output=response,
        )

        st.session_state[ID_CONVO] = convo


main()
