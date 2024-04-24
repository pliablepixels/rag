
import logging
from llm import MyLLM
from query import MyQueryProcessor

formatter = logging.Formatter(fmt= "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger('rag')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

my_llm = MyLLM()

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
    print ('\nFINAL ANSWER:\n')
    print(response)
    print ('\n---------------------------\n')
print ('\n\nGoodbye!\n\n')