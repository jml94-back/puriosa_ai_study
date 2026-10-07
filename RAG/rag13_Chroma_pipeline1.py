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
aaa = retriever.invoke(query)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0, #0:있는 그대로, 1:창의적으로
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

query_with_context=f"""
{aaa[0].page_content}\n\n
위 내용을 근거하여 다음 질문에 답변하세요.\n\n{query}
"""

response = model.invoke(query_with_context)
print(response.content)