# LCEL langchain Expression Language
# chain = prompt + model + output_parser  pmo
# parser: 분석하다, 답변을 다듬다

from langchain_openai  import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

template = "{topic}에 대해 쉽게 설명해줘."
ptem = PromptTemplate.from_template(template=template)

model = ChatOpenAI(
    model_name="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url,
    # openai_api_key=openai_api_key,
)

output_parser = StrOutputParser()

chain = ptem | model | output_parser

input = {"topic":"모든 activation 함수의 종류"}

response = chain.invoke(input)

print(response)
print(response.context)