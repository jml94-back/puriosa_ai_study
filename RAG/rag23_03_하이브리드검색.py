import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

######### BM25 검색 ###########
# Best Match. 키워드 기반 검색
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# for doc in bm25_docs:
#     print(doc.page_content)

######### 벡터 검색 ###########
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
    collection_name="chroma23",
)

verctor_retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)

# vector_docs = verctor_retriever.invoke(query)

# for doc in vector_docs:
#     print(doc.page_content)

#하이브리드 검색
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[verctor_retriever, bm25_retriever,],
    weights=[0.5, 0.5]
)

hybrid_docs = hybrid_retriever.invoke(query)


for doc in hybrid_docs:
    print(doc.page_content)