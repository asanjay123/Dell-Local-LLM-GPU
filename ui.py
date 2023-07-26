import os
os.environ["TRANSFORMERS_CACHE"] = "./models/transformers_cache"

import torch
import gradio as gr
import time
from langchain.llms.base import LLM
from langchain import OpenAI
from llama_index import (
    GPTListIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from transformers import pipeline

##################################### Utility Functions #####################################

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

##################################### Model Generation #####################################

max_token_count = 100
prompt_helper = PromptHelper(
    # maximum input size
    max_input_size=1024,
    # number of output tokens
    num_output=max_token_count,
    # the maximum overlap between chunks.
    max_chunk_overlap=20,
)

torch.cuda.set_per_process_memory_fraction(0.8, device=0)

class LocalOPT(LLM):
    # model_name = "facebook/opt-iml-max-30b" # (this is a 60gb model)
    model_name = "facebook/opt-iml-1.3b"  # ~2.63gb model -- limit on max tokens not reached
    # model_name = "gpt2"  # -- max input (file upload) tokens is 1024
    pipeline = pipeline("text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt: str, stop=None) -> str:
        response = self.pipeline(prompt, max_new_tokens=max_token_count)[0]["generated_text"]
        # only return newly generated tokens
        return response[len(prompt) :]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"

model_instance = LocalOPT()

##################################### Document Indexing & Chatbot Creation #####################################

@timeit()
def build_chat_bot():
    global index
    llm = LLMPredictor(llm=model_instance)
    service_context = ServiceContext.from_defaults(
       llm_predictor=llm, prompt_helper=prompt_helper
    )
    documents = SimpleDirectoryReader("db").load_data()
    index = GPTListIndex.from_documents(documents, service_context=service_context)
    print("Finished indexing")
    
    filename = "storage"
    print("Indexing documents...")
    index.storage_context.persist(persist_dir=f"./{filename}")
    storage_context = StorageContext.from_defaults(persist_dir=f"./{filename}")
    service_context = ServiceContext.from_defaults(llm_predictor=llm, prompt_helper=prompt_helper)
    index = load_index_from_storage(storage_context, service_context = service_context)
    print("Indexing complete")
    return('Index saved')

def chat(chat_history, user_input):
    print("Querying input...")
    query_engine = index.as_query_engine()
    print("Generating response...")
    bot_response = query_engine.query(user_input)

    response_stream = ""
    for letter in ''.join(bot_response.response):
        response_stream += letter + ""
        yield chat_history + [(user_input, response_stream)]
    
    print("Completed response generation")
    
##################################### File Upload & Store in Database #####################################

# Note: Uploaded Files must be more than 1KB in size or they are read as empty.

def copy_tmp_file(tmp_file, new_file):
    with open(tmp_file, "rb") as f:
        content = f.read()
    with open(new_file, "wb") as f:
        f.write(content)
    return None

def process_file(fileobj):
    script_dir = os.path.dirname(__file__)
    for obj in fileobj:
    # Store the file in the db directory (excludes repeats by name -- case insensitive).
        final_file_path = os.path.join(script_dir, "db", f"{os.path.basename(obj.name)}")
        copy_tmp_file(obj.name, final_file_path)

    print(final_file_path)
    return "Upload processed successfully"

##################################### Model Selection #####################################

def set_model(name):
    print(f"Loading model: {name}")
    # print(f"Token count from slider: {token_count}")

    if name != model_instance.model_name:
        model_instance.model_name = name
        model_instance.pipeline = pipeline("text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
    
    chatbot.label = model_instance.model_name
    build_chat_bot()

    print(f"Successfully loaded model: {name}")
    return (f"Successfully loaded model: {name}")
    
def get_models(directory):
    immediate_subdirectories = []
    for subdirectory in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdirectory)):
            immediate_subdirectories.append(subdirectory)
    return immediate_subdirectories

def show_models(directory):
    immediate_subdirectories = get_models(directory)
    dropdown_options = []
    for subdirectory in immediate_subdirectories:
        parts = subdirectory.split("models--")[1]
        model = parts.split("--")
        if len(model) > 1:
            owner = model[0]
            name = model[1]
            dropdown_options.append(f"{owner}/{name}")
        else:
            name = model[0]
            dropdown_options.append(f"{name}")
    
    dropdown = gr.Dropdown(dropdown_options, label="Pick a local model", interactive=True)
    return dropdown

def get_model_name():
    return model_instance.model_name

##################################### Gradio UI #####################################

with gr.Blocks() as demo:
    gr.Markdown('Chatbot Interface for Locally Hosted LLM')
    with gr.Tab("Database"):

        file_input = gr.File(file_count='multiple') # can set to 'single' or 'directory'
        upload_button = gr.Button("Upload")
        status = gr.Textbox(label="Status")
        upload_button.click(process_file, file_input, status)

    with gr.Tab("Model"):

        new_model = gr.Textbox(label="Paste a User/Model from HuggingFace")
        download_button = gr.Button("Download a New Model")

        model_picker = show_models("models/transformers_cache")
        load_model = gr.Button("Load Model")
        
        status = gr.Textbox(label="Status")

        download_button.click(set_model, new_model, status)
        load_model.click(set_model, model_picker, status)

    with gr.Tab("Attributes"): # In progress

        token_count = gr.Slider(minimum=10, maximum=1024, step=1, interactive=True, label="Max Token Count").value
        
        save_changes_button = gr.Button("Save Changes")
        
        status = gr.Textbox(label="Status")

    with gr.Tab("Chatbot"):

        chatbot = gr.Chatbot(label=f"{model_instance.model_name}")
        message = gr.Textbox(label="Input")
        message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch()
