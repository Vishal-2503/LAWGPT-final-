import os
import time
import streamlit as st
import asyncio

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set Streamlit page title
st.set_page_config(page_title="LawGPT")

# Fetch API Key safely
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

if not TOGETHER_API_KEY:
    st.error("⚠️ TOGETHER_API_KEY is missing! Please set it in your environment variables.")
    st.stop()

# Display Logo
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

# Apply Custom CSS
st.markdown(
    """
    <style>
    div.stButton > button:first-child { background-color: #ffd0d0; }
    div.stButton > button:active { background-color: #ff6262; }
    div[data-testid="stStatusWidget"] div button { display: none; }
    .reportview-container { margin-top: -2em; }
    #MainMenu { visibility: hidden; }
    .stDeployButton { display: none; }
    footer { visibility: hidden; }
    #stDecoration { display: none; }
    button[title="View fullscreen"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reset Chat
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load FAISS Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
)

try:
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)

    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# Define Prompt Template
prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question', 'chat_history']
)

# Initialize LLM (TogetherAI)
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_API_KEY
)

# Create Conversational Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Chat Input
input_prompt = st.chat_input("Ask a legal question...")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            try:
                result = qa.invoke(input=input_prompt)

                message_placeholder = st.empty()
                full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
                
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")
                
                message_placeholder.markdown(full_response)
                
                st.button('Reset Chat 🗑️', on_click=reset_conversation)

                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

# Handle AsyncIO Event Loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
