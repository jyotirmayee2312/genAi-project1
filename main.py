import os
import boto3
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.chat_models import BedrockChat

st.title("JyotiBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "./vector/index.pkl"


main_placeholder = st.empty()

bedrock_runtime = boto3.client( 
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    region_name="us-east-1",
    model_kwargs={"temperature": 0.0},
)
embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    )

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    vectorstore.save_local(folder_path="./vector", index_name='index')
    # retriever = vectorstore.as_retriever()
    
    # with open(file_path, "wb") as f:
    #     pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # with open(file_path, "rb") as f:
        #     # vectorstore = pickle.load(f)
        vectorstore = FAISS.load_local(folder_path="./vector", embeddings=embeddings,allow_dangerous_deserialization=True)
        # retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

            # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)




