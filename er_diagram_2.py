
import os
os.system("rm -rf qdrant_mm_db")  # else keeps adding to db

import qdrant_client
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms import OpenAI
from llama_index import ServiceContext

client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client,
    collection_name="text_collection",
)
image_store = QdrantVectorStore(
    client=client,
    collection_name="image_collection",
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store,
    image_store=image_store,
)

loader = SimpleDirectoryReader(
     input_files=["./images/bvbrc_er.jpg","./txt/83332.12_10lines.txt"])
documents = loader.load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

retriever_engine = index.as_retriever(
    similarity_top_k=3,
    image_similarity_top_k=3,
)

llm = OpenAI()
service_context = ServiceContext.from_defaults(llm=llm)

# results = retriever_engine.retrieve("Name a feature in the genome: Mycobacterium tuberculosis H37Rv.")
# for res_node in results:
    # print(res_node)

query_engine = index.as_query_engine(
    service_context=service_context,
    similarity_top_k=3,
    image_similarity_top_k=3,
)

# response = query_engine.query("Given the genome: Mycobacterium tuberculosis H37Rv, find one entity on the other side of the '")
response = query_engine.query("Given the feature: fig|83332.12.peg.1250, find the name of the entity on the other side of the 'Genome Feature' relationship.")
print(response)
