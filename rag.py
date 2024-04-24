
import logging
from llm import MyLLM
from query import MyQueryProcessor

formatter = logging.Formatter(fmt= "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger('rag')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#model_name="BAAI/bge-small-en-v1.5"
model_name="hkunlp/instructor-xl"
embed_model = HuggingFaceEmbedding(model_name=model_name)
#embed_model = HuggingFaceEmbedding(model_name="Salesforce/SFR-Embedding-Mistral")



my_llm = MyLLM()
#Settings.llm = my_llm

my_query = MyQueryProcessor(streaming=False, response_mode='refine')
my_query.load_or_create(path='./data', persist_dir='db')


print ('\n---------------------------')
while True:
    query = input("Enter query ( 'q' to quit)\n>> ")
    if query in ['exit', 'quit', 'q','e']:
        break
    logger.debug ('Thinking...\n')
    response = my_query.query(query)  
    ''' for token in response:
        print(token.delta, end="")
    #streaming_response.print_response_stream()'''
    print(response)
    print ('\n---------------------------\n')
print ('\n\nGoodbye!\n\n')