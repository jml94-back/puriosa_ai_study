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

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화를 작성해주세요.
양식은 [FROMAT]을 참고해주세요.
#상황:
{question}

#FORMAT:
-영어회화:
-한글번역:
"""
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

input = {"question":"저는 부산에서 밀면을 먹고 싶어요."}

response = chain.invoke(input)

print(response)