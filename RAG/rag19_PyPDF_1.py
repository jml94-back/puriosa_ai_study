import os

from langchain_classic.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"


path = "./_data/"
pdf_loader = PyPDFLoader(path+"attention_is_all_you_needs.pdf")

pdf_docs = pdf_loader.load()

print(type(pdf_docs))
print(len(pdf_docs))
# print(pdf_docs)

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ",""]
)

split_doc = pdf_loader.load_and_split(text_spliter)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=api_key,
#     base_url=base_url,
#     dimensions=5,
# )

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device":"cpu",
        "local_files_only":True,
    }
)

db = FAISS.from_documents(
    documents=split_doc,
    embedding=embeddings,
)

DB_PATH = "./_db/Pypdf"

db.save_local(
    folder_path=DB_PATH,
    index_name = "Pypdf1"
)