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

path = "./_data/rag_data/"
# loader1 = TextLoader(path+"2026_AI_for_All.txt")
# loader1 = TextLoader(path+"nvidia_outlook.txt", encoding="utf-8")
# loader2 = TextLoader(path+"samsung_outlook.txt", encoding="utf-8")

txt_files = glob (os.path.join(path,"*.txt"))

# print(txt_files) #['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']
data = []
for text_file in txt_files:
    loader = TextLoader(text_file,encoding="utf-8")
    data += loader.load()

# print(data)

# char_count = [len(doc.page_content) for doc in data]
# print(char_count)
# exit()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ",""]
)

texts = text_spliter.split_documents(data)
# print([len(doc.page_content) for doc in texts])
# print(texts[0])
# exit()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)

# sample_text = "삼성전자의 창업자는 누구인가요?"
# vector = embeddings.embed_query(sample_text)


DB_PATH = "./_db/Chroma12/"

vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12",
)
print(vector_store._collection.count())

query = "삼성전자 창업주"

########## Retrievers ###############
########## 벡터 검색기 ################
retriever = vector_store.as_retriever(search_kwarges={"k":2})
print(retriever)
aaa = retriever.invoke(query)
print(aaa)
print(aaa[0].page_content)
print(aaa[1].page_content)
print(aaa[2].page_content)