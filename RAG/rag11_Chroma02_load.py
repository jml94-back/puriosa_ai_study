import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# path = "./_data/rag_data/"
# # loader1 = TextLoader(path+"2026_AI_for_All.txt")
# loader1 = TextLoader(path+"nvidia_outlook.txt", encoding="utf-8")
# loader2 = TextLoader(path+"samsung_outlook.txt", encoding="utf-8")

# text_spliter = RecursiveCharacterTextSplitter(
#     chunk_size = 300,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", " ",""]
# )

# split_doc1 = loader1.load_and_split(text_spliter)
# split_doc2 = loader2.load_and_split(text_spliter)

# print(len(split_doc1)) #9
# print(len(split_doc2)) #9

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)


DB_PATH = "./_db/Chroma11"

db = Chroma(
    # documents=split_doc1+split_doc2,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma11",
)

print(db.get())
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2)
print(aaa)
# print("성공")