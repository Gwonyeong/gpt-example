from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files.
"""
)

class ChatCallback(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        

    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write('llm ended!')

    def on_llm_new_token(self, token: str, *args,  **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallback()]
)


@st.cache_data(show_spinner='Embedding file...')
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)

    if save:
        st.session_state['messages'].append({
            'message': message,
            'role': role
        })

def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], False)

def format_docs(docs):
    return "답변을 드리겠습니다. 😁 "+ "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    반드시 Context에서 찾아 대답을 해줘.
    만약, 네가 모르는 사실이 나온다면 말을 지어내지 말고 모른다고 말해.
     
     Context: {context}
     """
     ),
     ("human", "{question}")
])

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf .docx", type=["txt", "pdf", "docx"])



if file: 
    retriever = embed_file(file)
    paint_history()
    send_message(f"File {file.name} uploaded", "ai",False)
    message = st.chat_input("Ask a question")

    if message:
        send_message(message, 'human', True)
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message('ai'):
            res =  chain.invoke(message)
            
else:
    st.session_state['messages'] = []
