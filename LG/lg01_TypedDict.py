from typing import Dict,TypedDict

sample_dict : Dict[str,str] = {
    "name":"김말똥",
    "age": 39,
    "job": "회사원",
}
print(sample_dict)

class Person(TypedDict):
    name : str
    age: int
    job: str

p1 = Person( name="길동",age="40",job="도적",)
print(p1)

p2 : Person = {"name": "개똥","age":20,"job":"전사"}
print(p2)