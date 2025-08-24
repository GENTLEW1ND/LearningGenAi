from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

# load env variables first
load_dotenv()
# mytoken = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_huggingai_response(question):
    # create Hugging Face model after env is loaded
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-v0.1",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=200,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    return llm(question)

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
