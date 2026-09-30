from langchain_core.prompts import PromptTemplate

template = "{country}의 수도는 어디인가요?"

ptem = PromptTemplate.from_template(template=template)
print(ptem) #input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'