import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.document_loaders import PyPDFLoader

filename = "ReAct.pdf"
loader = PyPDFLoader(filename)
elements = loader.load_and_split()

documents = []
ids = []

for element in elements:
    documents.append({"content": str(element), "metadata": {"filename": filename}})
    ids.append("content_" + str(len(documents) - 1))
    
print(len(documents))
print(documents[:10])

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chromadb_data_2"
))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)

collection = client.create_collection(name="langchain", embedding_function=openai_ef)

chunk_size = 100
for i in range(0, len(documents), chunk_size):
    chunk_docs = documents[i:i + chunk_size]
    chunk_ids = ids[i:i + chunk_size]
    content_list = [d['content'] for d in chunk_docs]
    metadata_list = [d['metadata'] for d in chunk_docs]
    print(f"Adding chunk: {i // chunk_size + 1}")
    collection.add(documents=content_list, metadatas=metadata_list, ids=chunk_ids)

client.persist()

print(client.get_collection("langchain", openai_ef))