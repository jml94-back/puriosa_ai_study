import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

#pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

path = "./_data/rag_data/"
# loader1 = TextLoader(path+"2026_AI_for_All.txt")
loader1 = TextLoader(path+"nvidia_outlook.txt", encoding="utf-8")
loader2 = TextLoader(path+"samsung_outlook.txt", encoding="utf-8")

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ",""]
)

split_doc1 = loader1.load_and_split(text_spliter)
split_doc2 = loader2.load_and_split(text_spliter)

print(len(split_doc1)) #9
print(len(split_doc2)) #9

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)

############ FAISS
# faiss_index = faiss.IndexFlatL2(5)#len(embeddings.embed_query("")))
# print("FAISS 인덱스 초기화 준비 완료")

# #임베딩 차원 수
# print(faiss_index.d)

# faiss_db = FAISS(
#     embedding_function=embeddings,
#     index=faiss_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},

# )

# print(faiss_db.index.ntotal)

########## from langchain
db = FAISS.from_documents(
    documents=split_doc1+split_doc2,
    embedding=embeddings,
)

DB_PATH = "./_db/FAISS17"

db.save_local(
    folder_path=DB_PATH,
    index_name = "faiss17"
)