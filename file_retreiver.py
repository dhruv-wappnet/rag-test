from Collection.upload_files import get_connection_string
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import time

embedding_model = HuggingFaceEmbeddings()
db = PGVector(
    embedding_function=embedding_model,
    collection_name="test-collection",
    connection_string=get_connection_string()
)

retriever = db.as_retriever(
    # search_kwargs={"k": 5,"score_threshold": 0.5},
    # search_type="similarity_score_threshold"
)

# import llama2 for context based search from hggingface
# Load model directly

ckpt = 'deepset/tinyroberta-squad2'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(ckpt)
# Use a pipeline as a high-level helper

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

instruction = """
    You are a question answering bot. You will be passed a context for generating response. You will be given context. Answer the question based on the context.
    Be sure to generate the correct response. If context does not contain the answer, you can respond with "I don't know"
"""
# tracking retreiaval time
question = input("Enter the question: ")
start = time.time()
context_list = retriever.invoke(question, k=3)
context = context_list[0].page_content + context_list[1].page_content + context_list[2].page_content
# print(context)
query = f"""{instruction}
    Question : {question}"""

QA_input = {
    'question': query,
    'context': context
}

response = pipe(QA_input)
print(response['answer'])

end = time.time()
print(f"Retrieval time : {end - start} seconds")

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