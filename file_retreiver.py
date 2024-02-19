from Collection.upload_files import get_connection_string
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

embedding_model = HuggingFaceEmbeddings()
db = PGVector(
    embedding_function=embedding_model,
    collection_name="test-collection",
    connection_string=get_connection_string()
)

retriever = db.as_retriever(
    search_kwargs={"k": 5,"score_threshold": 0.5},
    search_type="similarity_score_threshold"
)

# import llama2 for context based search from hggingface
# Load model directly

ckpt = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)
# Use a pipeline as a high-level helper

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

instruction = """
    You are a question answering bot. You will be passed a context for generating response. You will be given context in format:
    Context : "This is a context"
    Question : "What is the context?"
    Be sure to generate the correct response. If context does not contain the answer, you can respond with "I don't know"
"""
# tracking retreiaval time
question = input("Enter the question: ")
start = time.time()
context = retriever.invoke(question)
end = time.time()
print(f"Retrieval time : {end - start} seconds")

query = f"""{instruction}
    Context : {context[0].page_content + context[1].page_content}
    Question : {question}"""

response = pipe(question=question)
print(response)
# def main():
#     retriever = db.as_retriever(
#         search_kwargs={"k": 5,"score_threshold": 0.5},
#         search_type="similarity_score_threshold"
#     )
#     query_string = 'What types of AI technologies does WappnetAI specialize in?'
#     result = retriever.invoke(query_string)

#     print(result[0].page_content)
#     from transformers import pipeline
#     hf_model = "distilgpt2"
#     tokenizer = "distilgpt2"
#     qa = pipeline("question-answering", model= hf_model, tokenizer=tokenizer)
#     answer = qa(question=query_string, context=result[0].page_content + result[1].page_content)
#     print(answer)