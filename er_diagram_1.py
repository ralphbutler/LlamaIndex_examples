
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index import SimpleDirectoryReader

image_documents = SimpleDirectoryReader(input_files=["./images/bvbrc_er.jpg"]).load_data()

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=1500
)

descr = """
The image contains an ER diagram where Entities are depicted by squares
and Relationships are depicted by diamonds.
"""

query = "If you were to describe the layout of the ER diagram, in what order would you list the nodes?"

response_1 = openai_mm_llm.complete(
    prompt=f"{descr}{query}",
    image_documents=image_documents,
)

print("RESPONSE_1",response_1)
