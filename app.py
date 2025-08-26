import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
import huggingface_hub
import streamlit as st

# Load env variables (optional if storing token in .env)
load_dotenv(".env")

# Use the new fine-grained token here
mytoken = os.getenv("HUGGINGFACEHUB_API")

def get_huggingai_response(question):
     # Force Hugging Face to use your token explicitly
    huggingface_hub.login(token=mytoken)
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b", 
        task="conversational",
        temperature=0.7,
        max_new_tokens=100,
        huggingfacehub_api_token=mytoken
    )
    chatmodel = ChatHuggingFace(llm=llm)
    response = chatmodel.invoke(question)
    return response

st.set_page_config(page_title="Q&A Demo")
st.header("Personal GPT")

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input:
    with st.spinner("Generating response..."):
        try:
            response = get_huggingai_response(input)
            st.subheader("The Response Is")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API token and model availability")
