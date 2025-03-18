import os
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever

def process_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return  chunks
def get_chain(chunks):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    # file_path = r"Article_93.pdf"

    # # Load the PDF
    # loader = PDFPlumberLoader(file_path)
    # documents = loader.load()

    # # Split into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    # chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store (in-memory)
    embedding_function = OllamaEmbeddings(model="llama3")
    vector_db = FAISS.from_documents(chunks, embedding_function)

#     # read vector base locally
#     vector_db = Chroma(
#     persist_directory="./chroma_db",  # 指定本地数据库目录
#     embedding_function=OllamaEmbeddings(model="llama3"),  # 指定嵌入模型
#     collection_name="local-rag"  # 指定集合名称
# )

    # LLM from Ollama
    local_model = "llama3"
    llm = ChatOllama(model=local_model)

    # MultiQueryRetriever Prompt
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate five alternative versions of the given question 
        to improve document retrieval from a vector database.
        Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG Prompt
    template = """Using the provided context, answer the question comprehensively.
    Respond only with relevant and concise information.
    If the answer cannot be found in the context, do not make assumptions.
    Context: {context}
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain