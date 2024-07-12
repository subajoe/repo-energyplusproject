from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_objectbox import ObjectBox
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = ObjectBox.from_documents(document_chunks, OllamaEmbeddings(), embedding_dimensions=786)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = Ollama(model="llama2")
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to "
         "the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_response(user_input, vectorDB, chathistory):
    retriever_chain = get_context_retriever_chain(vectorDB)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": chathistory,
        "input": user_input
    })

    return response['answer']


def get_conversational_rag_chain(retriever_chain):
    llm = Ollama(model="llama2")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
