#tool calling : 외부 기능을 가져와서 사용하기

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

llm = ChatOpenAI(
    # model_name="gpt-5-nano",
    model_name="gpt-5.6-terra",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# pip install langchain_tavily
# from langchain_tavily import TavilySearch

# web_search = TavilySearch(
#     max_result = 3,
# )
# result = web_search.invoke({"query":"삼성전자의 오늘 주가"})
# print(result["results"][-1]["content"])

# pip install -U ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
web_search = DuckDuckGoSearchRun( max_result = 5)
# result = web_search.invoke({"query":"삼성전자의 오늘 주가"})
# print(result)
# # print(result["results"][-1]["content"])
# exit()

tools=[web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition

######## 1. State 정의
class State(TypedDict):
    messages : Annotated[list,add_messages]

######## 2. node 정의
def chatbot(state : State):
    return {
        "messages":[llm_with_tools.invoke(state["messages"])]
    }

######## 3. 그래프 구성
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot",chatbot)

tool_node = ToolNode(tools)
graph_builder.add_node("tools",tool_node)

######## 4. 엣지 추가
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition, #tool_calls 있는지 판단                           
    #tool_calls 있으면 tools 반환
    #tool_calls 없으면 END 반환
    #tools_condition에 END 포함
)
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge("chatbot", END) 필요 없음.

######## 5. 그래프 컴파일
graph = graph_builder.compile()

######## 6. 그래프 실행
query = "현재 필리핀 대통령이 누구야?"

response = graph.invoke(
    {
        "messages":[("user",query)]
    }
)

######## 7. 결과 출력

for i, msg in enumerate(response["messages"]):
    print(f"[{i}] {msg.type} : {msg.content}")