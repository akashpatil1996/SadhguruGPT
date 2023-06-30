from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings



OPENAI_API = st.secrets["OPENAI_API"]
PINECONE_API = st.secrets["PINECONE_API"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API, model='text-embedding-ada-002')
index_name = "for-langchain"
pinecone.init(api_key=PINECONE_API , environment=PINECONE_ENV )
index = pinecone.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API, model='text-embedding-ada-002')

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Sadhguru: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

def find_match(input):
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    result = vectorstore.similarity_search(input, k=2)
    return result



st.markdown("<h1 style='text-align: center;'>SadhguruGPT 🧘</h1>", unsafe_allow_html=True)


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Namaskaram🙏 Please ask your question"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are Sadhguru, a yogi, mystic and a spiritual guru, Now answer the question with only the context provided and as truthful as possible, and if the answer is not contained within the text below then respond 'I don't know'. If asked 'how are you' or something similar then reply "I'm doing good, Please ask your question". If you are greeted then greet them back. Context: """)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Loading..."):
            conversation_string = get_conversation_string()
            context = find_match(query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
