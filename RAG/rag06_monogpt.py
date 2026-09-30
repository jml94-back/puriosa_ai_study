from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

llm = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
    # openai_api_key=openai_api_key,
)

response = llm.invoke("안녕하세요")
# print(response)
print(response.content)