import sys
import langchain
print('Python:', sys.version)
print('LangChain:', langchain.__version__)
try:
    import ragas
    print('RAGAS:', ragas.__version__)
except ImportError:
    print('RAGAS: Not installed')
