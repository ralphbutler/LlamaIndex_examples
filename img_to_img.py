
import os
os.system("rm -rf qdrant_img_db")  # else keeps adding to db

from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext

import qdrant_client
from llama_index import SimpleDirectoryReader

client = qdrant_client.QdrantClient(path="qdrant_img_db")

text_store = QdrantVectorStore(
    client=client,
    collection_name="text_collection",
)
image_store = QdrantVectorStore(
    client=client,
    collection_name="image_collection",
)
storage_context = StorageContext.from_defaults(vector_store=text_store,
                                               image_store=image_store)

# Create the MultiModal index
documents = SimpleDirectoryReader("./images/").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

retriever_engine = index.as_retriever(image_similarity_top_k=3)
retrieval_results = retriever_engine.image_to_image_retrieve(
    "./dblhelix_for_query.png"
)
retrieved_images = []
for res in retrieval_results:
    retrieved_images.append(res.node.metadata["file_path"])
    print("DBG",res,res.node.metadata["file_path"])

#### code to display the selected images
from PIL import Image
import matplotlib.pyplot as plt
import os

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if ".jpg" in img_path  or  ".png" in img_path:
            image = Image.open(img_path)
            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 9:
                break
    return plt

# print("DBGLEN",len(retrieved_images))
# for img in retrieved_images:
    # print("  ",img)
plt = plot_images(retrieved_images)
plt.show()

# print("EXITING ****") ; exit(0)

from llama_index.prompts import PromptTemplate
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=1500,
)

qa_template_str = (
    "Given the images provided, respond to the prompt.\n"
    "Prompt: {query_str}\n"
    "Response: "
)

qa_template = PromptTemplate(qa_template_str)

query_engine = index.as_query_engine(
    multi_modal_llm=openai_mm_llm, image_qa_template=qa_template,
    similarity_top_k=3, image_similarity_top_k=3
)

query = "Tell me about some biology-related images like this one."
text_response = query_engine.image_query("./dblhelix_for_query.png", query)
print("\n\nTEXTRESP",text_response)
