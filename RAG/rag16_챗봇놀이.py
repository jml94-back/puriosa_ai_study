import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from glob import glob

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)

DB_PATH = "./_db/Chroma12/"

vector_store = Chroma(
    # documents=texts,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12",
)

query = "삼성전자의 창업주는 누구인가요?"
retriever = vector_store.as_retriever(search_kwarges={"k":4})
# aaa = retriever.invoke(query)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0, #0:있는 그대로, 1:창작 허용
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# query_with_context = f"""
# {aaa[0].page_content}\n\n
# 위 내용을 근거하여 다음 질문에 답변하세요.\n\n{query}
# """

# response = model.invoke(query_with_context)
# print(response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답해주세요. 컨텍스트 관련 정보가 없다면, 
"주어진 정보로는 답변할 수 없습니다." 라고 말해주세요.

컨텍스트:{context}
질문:{input}
답변:
""")

#체인 만들기
docu_chain = create_stuff_documents_chain(model,prompt) #모델과 프롬프트 연결  prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain)
"""
#체인 실행
response = rag_chain.invoke({"input":query})

print(response)
print(response.keys())
"""
###################### Gradio 챗봇 ########################

import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input":message}) 
    return response["answer"]

# Gradio 인터페이스

demo = gr.ChatInterface(fn = answer_invoke, title="데모봇!!")

demo.launch()