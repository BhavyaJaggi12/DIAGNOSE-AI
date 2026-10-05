from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print(llm.invoke("hello").content)
    print("gemini-2.5-flash works")
except Exception as e:
    print("gemini-2.5-flash failed:", e)
