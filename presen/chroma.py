from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI


# loader = PyPDFLoader("ReAct.pdf")
# documents = loader.load_and_split()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# embedding = OpenAIEmbeddings()

# persist_directory = 'db'
persist_directory = 'chromadb_data'

embedding = OpenAIEmbeddings()
# vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)

# vectordb.persist()
# vectordb = None

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

docs = vectordb.similarity_search("線形回帰", k=3, return_only_outputs=True)

# print(docs)

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
query = "線形回帰とはなんでしょうか。"
print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))