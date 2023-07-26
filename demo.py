"""
Optional: Change where pretrained models from huggingface will be downloaded (cached) to:
export TRANSFORMERS_CACHE=/whatever/path/you/want
"""

import os
os.environ["TRANSFORMERS_CACHE"] = "./models/transformers_cache"

import time
from langchain.llms.base import LLM
from llama_index import (
    GPTListIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Document
)
from transformers import pipeline
import gradio as gr

def timeit():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]

            print(f"[{(end - start):.8f} seconds]")
            return result

        return wrapper

    return decorator

prompt_helper = PromptHelper(
    # maximum input size
    max_input_size=2048,
    # number of output tokens
    num_output=1024,
    # the maximum overlap between chunks.
    max_chunk_overlap=100,
)


class LocalOPT(LLM):
    # model_name = "facebook/opt-iml-max-30b" # (this is a 60gb model)
    model_name = "facebook/opt-iml-1.3b"  # ~2.63gb model
    pipeline = pipeline("text-generation", model=model_name)

    def _call(self, prompt: str, stop=None) -> str:
        response = self.pipeline(prompt, max_new_tokens=1024)[0]["generated_text"]
        # only return newly generated tokens
        return response[len(prompt) :]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"


@timeit()
def create_index():
    print("Creating index...")
    # Wrapper around an LLMChain from Langchaim
    llm = LLMPredictor(llm=LocalOPT())
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm, prompt_helper=prompt_helper
    )
    docs = SimpleDirectoryReader("db").load_data()
    index = GPTListIndex.from_documents(docs, service_context=service_context)
    print("Done creating index", index)
    return index

@timeit()
def execute_query(input):
    query_engine = index.as_query_engine()
    response = query_engine.query(input)
    return response

if __name__ == "__main__":
    
    filename = "storage"
    if not os.path.exists(filename):
        print("No cached index found")
        index = create_index()
        index.storage_context.persist(persist_dir=f"./{filename}")
    else:
        print("Loading local cache of model...")
        llm = LLMPredictor(llm=LocalOPT())
        storage_context = StorageContext.from_defaults(persist_dir=f"./{filename}")
        service_context = ServiceContext.from_defaults(llm_predictor=llm, prompt_helper=prompt_helper)
        index = load_index_from_storage(storage_context, service_context = service_context)
        print("Finished loading locally cached model")

    print("Executing queried input...")
    response = execute_query("What does redfish enable?")
    print(response)
    print("Finished executing queried input")

