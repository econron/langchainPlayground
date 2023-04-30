import tempfile
import json
import os
from unstructured.partition.md import partition_md
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def partition_ipynb(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    elements = []
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            elements.append("```python\n" + "".join(cell["source"]) + "\n```\n")
        elif cell["cell_type"] == "markdown":
            elements.append("".join(cell["source"]) + "\n")

    text = "\n".join(elements)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=True) as f:
        f.write(text)
        f.flush()
        result = partition_md(filename=f.name)

    return result

directory_path = './data'
file_list = os.listdir(directory_path)
documents = []
ids = []
for filename in file_list:
    file_path = os.path.join(directory_path, filename)
    if file_path.endswith(".ipynb"):
        elements = partition_ipynb(file_path)
    elif file_path.endswith(".md"):
        elements = partition(file_path)
    elif file_path.endswith(".rst"):
        elements = partition(file_path)
    else:
        elements = []
    for element in elements:
        documents.append({"content": str(element), "metadata": {"filename": filename}})
        ids.append("content_" + str(len(documents) - 1))

print(len(documents))
print(documents[:2])
print(len(ids))
print(ids[:2])


client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chromadb_data"
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