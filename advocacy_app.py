import os
from langchain.vectorstores import Chroma
import streamlit.components.v1 as components
import streamlit as st
import sys

# this is a workaround because of an outdated version
# pf sqlite on streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def load_pdf(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    else:
        print('Only PDF files supported!')
        return None

    data = loader.load()
    return data

# chunk the document


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings in chroma db
# only run this once in development because
# final app will always use the same DB


def create_embeddings(chunks, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            chunks, embeddings, persist_directory=directory)
        return vector_store


def load_embeddings(directory):
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=directory,
                          embedding_function=embeddings)
    return vector_store

# get answer from chatGPT, increase k for more elaborate answers


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={'k': k}, )
    prompt_template = """You are a UK mental health solicitor looking at sections of the the mental health act and code of practice. Use the following piece of context to answer the questions at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer If any sections of legislation are mentioned in the context (e.g. Section 28) include them in your answer with "See: section x, section y...".

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    answer = chain.run(q)
    return answer


# Only run on initialisation
# Loads, chunks and embedds pdf into a new chroma db
def create_new_vector_store(pdf, db_directory):
    document = load_pdf(pdf)
    chunks = chunk_data(document)
    vector_store = create_embeddings(chunks, db_directory)
    return vector_store


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # Check if OPENAI_API_KEY is set in .env
    if os.getenv('OPENAI_API_KEY') is None:
        # If not, retrieve it from Streamlit's secrets and set it as an environment variable
        os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

    st.image("images/ai-mental-health-expert.jpg")
    st.subheader("Answering questions based on the MHA")

    # Load the vector stores if we don't have them
    if 'mha_vector_store' not in st.session_state:
        with st.spinner("Loading The Mental Health Act"):
            st.session_state.mha_vector_store = load_embeddings("mha_db")

    if 'cop_vector_store' not in st.session_state:
        with st.spinner("Loading The Mental Health Act"):
            st.session_state.cop_vector_store = load_embeddings("cop_db")

    st.subheader("Enter Your Query:")
    with st.form(key='query_form'):
        query_input = st.text_input("Query:")
        submit_button = st.form_submit_button(label='Submit Query')

    if submit_button:
        with st.spinner("Thinking... "):
            answer_mha = ask_and_get_answer(
                st.session_state.mha_vector_store, query_input, k=9)
            answer_cop = ask_and_get_answer(
                st.session_state.cop_vector_store, query_input, k=9)
        answer = f"From The Mental Health Act \n\n {answer_mha} \n\n From The Code Of Practice \n\n {answer_cop}"
        st.write(answer)
