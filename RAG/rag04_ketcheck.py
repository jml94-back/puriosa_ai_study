import os
key = os.getenv("OPENAI_API_KEY")

if key is None:
    print("Empty")
else:
    print("키길이:",len(key))
    print("키확인:",key[:8],"...",key[-4:])