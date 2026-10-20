from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

llm = ChatOpenAI(
    model_name="gpt-5-nano",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

######## 1. State 정의
class State(TypedDict):
    messages : Annotated[list,add_messages]
    aaa : str
    bbb : str
    ccc : int

######## 2. node 정의
def chatbot(state: State):
    return {"message":[llm.invoke(state["messages"])]}

######## 3. 그래프 구성
graph_01 = StateGraph(State)
graph_01.add_node("chatbot",chatbot)

######## 4. 엣지 추가
graph_01.add_edge(START, "chatbot", )
graph_01.add_edge("chatbot", END)

######## 5. 그래프 컴파일
graph = graph_01.compile()

######## 6. 그래프 실행
query = "상성전자 예상 주가"

response = graph.invoke(
    {
        "messages":[("user",query)]
    }
)

print(response["messages"][1].content)
print(response["messages"][0].type)
print(response["messages"][1].type)

