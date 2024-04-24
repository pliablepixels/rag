from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage # type: ignore

import logging

class MyQueryProcessor:
    def __init__(self, streaming=False, response_mode='refine'):
        logger = logging.getLogger('rag')
        model_name="hkunlp/instructor-xl"
        logger.debug(f'Loading embedding:{model_name}')
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        Settings.embed_model = embed_model
        self.query_engine = None
        self.streaming = streaming
        self.response_mode = response_mode

        MISTRAL_7B_QA_PROMPT_TMPL = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and no prior knowledge, "
        "answer the query. Always mention the sources of your information.\n"
        "Query: {query_str}\n"
        "Answer: "
        )

        MISTRAL_7B_REFINE_PROMPT_TMPL = (
            "The original query is as follows: {query_str}\n"
            "We have provided an existing answer: {existing_answer}\n"
            "We have the opportunity to refine the existing answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the query. "
            "If the context isn't useful, return the original answer. Always mention the sources of your information.\n"
            "Refined Answer: "
        )
        self.MISTRAL_7B_QA_PROMPT = PromptTemplate(
            MISTRAL_7B_QA_PROMPT_TMPL
        )
        self.MISTRAL_7B_REFINE_PROMPT = PromptTemplate(
            MISTRAL_7B_REFINE_PROMPT_TMPL
        )     

    def load_or_create(self,path='./data', persist_dir='db'):
        logger = logging.getLogger('rag')
        logger.debug (f'Loading index from storage if possible with persist_dir:{persist_dir} and path:{path}')
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.debug ('Loading data and creating index')
            reader = SimpleDirectoryReader(input_dir=path, filename_as_id=True, recursive=True)
            docs = []
            for doc in reader.iter_data():
                logger.debug (f"Processed: {doc[0].metadata['file_path']}")
                docs.extend(doc)
            index = VectorStoreIndex.from_documents(docs)
            index.storage_context.persist(persist_dir)
        self.query_engine = index.as_query_engine(streaming=self.streaming, response_mode=self.response_mode,  
            text_qa_template=self.MISTRAL_7B_QA_PROMPT , refine_template=self.MISTRAL_7B_REFINE_PROMPT)

    def query(self, query_str):
        logger = logging.getLogger('rag')
        logger.debug (f'Querying with query_str:{query_str}')
        return self.query_engine.query(query_str)
    
   