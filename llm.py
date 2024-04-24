
from typing import Optional, List, Mapping, Any
#
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings

import logging


class MyLLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 256
    model_name: str = "Mistral-7B"
    llm: LlamaCPP = None
    invocation_count: int = 1

    def __init__(self,  model_path=None, temperature=None, context_window=None, 
                 generate_kwargs=None, model_kwargs=None, max_tokens=None, verbose=None, num_output=None):
        super().__init__()

        logger = logging.getLogger('rag')

        model_path = model_path or '/Users/arjun/fiddle/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf'
        temperature = temperature or 0.9
        context_window = context_window or 4096
        max_tokens = max_tokens or 1024
        generate_kwargs = generate_kwargs or {}
        verbose = verbose  or False
        model_kwargs = model_kwargs or {"n_gpu_layers": -1, "n_batch": 512}
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=verbose,
        )
        Settings.llm = self

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger = logging.getLogger('rag')
        logger.debug (f'============= Invocation:#{self.invocation_count}======================')
        self.invocation_count = self.invocation_count + 1
        logger.debug(f"PROMPT: {prompt} kwargs: {kwargs}")
        txt =self.llm.complete(prompt, **kwargs)
        #logger.debug ('----------------------------------')
        #logger.debug(f"RESPONSE: {txt}")
        #logger.debug ('----------------------------------')

        return CompletionResponse(text=str(txt))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:

        raise NotImplementedError("Streaming not supported")
        logger = logging.getLogger('rag')
        logger.debug (f'============= Streaming Invocation:#{self.invocation_count}======================')
        self.invocation_count = self.invocation_count + 1
        logger.debug(f"PROMPT: {prompt} kwargs: {kwargs}")
        response = ""
        response_iter = self.llm.stream_complete(prompt, **kwargs)

        response = ""
        for token in response_iter:
            response += token
            yield CompletionResponse(text=response, delta=token)

       



            
