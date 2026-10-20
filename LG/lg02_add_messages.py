from typing import Annotated,TypedDict
from langgraph.graph import add_messages

class MyData(TypedDict):
    messages : Annotated[list, add_messages] #Reducer

from langchain_core.messages import AIMessage,HumanMessage

msg1 = [HumanMessage(content="안녕",id="1")]
msg2 = [AIMessage(content="방가",id="2")]

results = add_messages(msg1, msg2)
print(results)

