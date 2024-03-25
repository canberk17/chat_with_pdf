import json
import os
import sys
import boto3
import streamlit as st

# Titan Embeddings Model to generate Embedding

from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.llms.bedrock import Bedrock


## Data Ingestion


import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

#Vector Embedding
from langchain_community.vectorstores import FAISS

##LLM Models

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pathlib import Path


bedrock=boto3.client(service_name="bedrock-runtime")

bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

data_directory = "data"
Path(data_directory).mkdir(parents=True, exist_ok=True)

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return len(uploaded_files)

def data_ingestion():
    loader = PyPDFDirectoryLoader(data_directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store


def get_vector_store(docs):
  vectorstore_faiss=FAISS.from_documents(
    docs,
    bedrock_embeddings
  )

  vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
  ###Create the Anthropic Model
  llm=Bedrock(model_id="anthropic.claude-v2",client=bedrock)
  return llm

def get_llama2_llm():
  ###Create the Anthropic Model
  llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
              model_kwargs={'max_gen_len':512})
  return llm


prompt_template="""
Human: Use the following pieces of context to provide concise answer to the question at the end but use at least 250 words with detailed explnations to summarize. If you dont know the answer, just say that you don't know, don't try to make up an answer 
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
  template=prompt_template,input_variables=["context","question"]
)


def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
   st.set_page_config("Chat PDF")
   st.header("Chat with PDF using AWS Bedrock 🛏️ 🪨 ")

   user_question=st.text_input("Ask a Question from the PDF Files")

   with st.sidebar:
      # File uploader
      uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
      if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        st.success(f"Saved {saved_files} files to {data_directory}")

      st.title("Update or Create Vector Store:")

      if st.button("Vectors Update"):
         with st.spinner("Processing..."):
          docs=data_ingestion()
          get_vector_store(docs)
          st.success('Done')

   if st.button("Claude Output"):
      with st.spinner("Processing..."):
        faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
        llm=get_claude_llm()

         #faiss_index=get_vector_store(docs)
        st.write(get_response_llm(llm,faiss_index,user_question))
        st.success('Done')

   if st.button("Llama2 Output"):
      with st.spinner("Processing..."):
        faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
        llm=get_llama2_llm()

         #faiss_index=get_vector_store(docs)
        st.write(get_response_llm(llm,faiss_index,user_question))
        st.success('Done')

if __name__=="__main__":
  main()