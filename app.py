from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import os

# load env variables first
load_dotenv(".env")
mytoken = "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"


def get_huggingai_response(question):
    # create Hugging Face model after env is loaded
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=200,
    )
    chatmodel = ChatHuggingFace(llm=llm)
    response = chatmodel.invoke(question)
    return response

st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input=st.text_input("Input: ", key="input")
submit=st.button("Ask the question")

if submit and input:
    with st.spinner("Generating response..."):
        try:
            response = get_huggingai_response(input)
            st.subheader("The Response Is")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API token and model availability")
