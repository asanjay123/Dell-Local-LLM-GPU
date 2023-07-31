import os
os.environ["TRANSFORMERS_CACHE"] = "./models/transformers_cache"

import torch
from torch.utils.data import Dataset, DataLoader
import gradio as gr
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
)
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
# from concurrent.futures import ThreadPoolExecutor



##################################### Parallelization #####################################
    
def batch_process(user_inputs):
    # Ensure user_inputs is a list
    if not isinstance(user_inputs, list):
        user_inputs = [user_inputs]

    # Tokenization
    inputs = tokenizer(user_inputs, return_tensors='pt', padding=True, truncation=True, max_length=prompt_helper.max_input_size)
    inputs = {key: val.to("cuda:0") for key, val in inputs.items()}  # Move the input tensors to GPU

    # Model Inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Decoding
    generated_text_ids = logits.argmax(-1)
    generated_texts = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)

    return generated_texts


def chat(chat_history, user_inputs):
    user_inputs = [prompt_helper.preprocess_input(prompt) for prompt in user_inputs]

    # Process the user inputs in parallel using batch_process
    generated_responses = batch_process(user_inputs)

    for user_input, response in zip(user_inputs, generated_responses):
        yield chat_history + [(user_input, response)]

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
    ###pipeline = pipeline("text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

    # def _call(self, prompt: str, stop=None) -> str:
    #     response = self.pipeline(prompt, max_new_tokens=max_token_count)[0]["generated_text"]
    #     # only return newly generated tokens
    #     return response[len(prompt) :]

    def _call(self, prompt: str, stop=None) -> str:
        response = batch_process(prompt)[0]
        return response[len(prompt):]

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

# def chat(chat_history, user_input):
#     print("Querying input...")
#     query_engine = index.as_query_engine()
#     print("Generating response...")
#     bot_response = query_engine.query(user_input)

#     response_stream = ""
#     for letter in ''.join(bot_response.response):
#         response_stream += letter + ""
#         yield chat_history + [(user_input, response_stream)]
    
#     print("Completed response generation")
    
##################################### File Upload & Store in Database #####################################

# Note: Uploaded Files must be more than 1KB in size or they are read as empty.

def copy_tmp_file(tmp_file, new_file):
    with open(tmp_file, "rb") as f:
        content = f.read()
    with open(new_file, "wb") as f:
        f.write(content)
    return None

def list_files(directory):
    files = os.listdir(directory)
    file_list = []
    for filename in files:
        file_path = os.path.join(directory, filename)
        file_size = os.path.getsize(file_path)
        file_list.append((filename, file_size))
    return file_list

def process_file(fileobj):
    script_dir = os.path.dirname(__file__)
    for obj in fileobj:
    # Store the file in the db directory (excludes repeats by name -- case insensitive).
        final_file_path = os.path.join(script_dir, "db", f"{os.path.basename(obj.name)}")
        copy_tmp_file(obj.name, final_file_path)

    print(final_file_path)
    return list_files("db")

##################################### Model Selection #####################################

def set_model(name):
    global tokenizer, model 
    print(f"Loading model: {name}")
    
    if name != model_instance.model_name:
        model_instance.model_name = name
        model_instance.pipeline = pipeline("text-generation", model=name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
    
        # Instantiate the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_instance.model_name)
        model = AutoModelForCausalLM.from_pretrained(model_instance.model_name).to("cuda:0")
    
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

database_interface = gr.Interface(
    fn=process_file,
    inputs=[gr.File(label="Upload files", file_count="multiple")],
    outputs=[gr.DataFrame(
        list_files("db"),
        headers=["File Name", "File Size (bytes)"],
        datatype=["str", "number"],
        max_cols=(2),
        label="Database",
    )]
)

def set_model_with_input(new_model, model_picker):
    if new_model:
        return set_model(new_model)
    elif model_picker:
        return set_model(model_picker)
    else:
        raise ValueError("Invalid input: both textbox upload and dropdown selection are empty.")

model_selection_interface = gr.Interface(
    fn=set_model_with_input,
    inputs=[gr.Textbox(label="Paste a User/Model from HuggingFace"), show_models("models/transformers_cache")],
    outputs=[gr.Textbox(label="Status")]
)

def adjust_attributes(token_count, overlap):
    global max_token_count
    global max_overlap
    
    max_token_count = token_count
    max_overlap = overlap

    attribute_summary = f"max_token_count = {max_token_count}, max_overlap = {max_overlap}"
    return attribute_summary

attributes_interface = gr.Interface(
    fn=adjust_attributes,
    inputs=[gr.Slider(value=100, minimum=10, maximum=1024, step=1, interactive=True, label="Max output tokens"),
            gr.Slider(value=20, minimum=0, maximum=max_token_count, step=1, interactive=True, label="Max chunk overlap")
            ],
    outputs=[gr.Textbox(label="Status")]
)

chat_interface = gr.ChatInterface(
    fn=chat,
    chatbot = gr.Chatbot(label="Chatbot"),
    textbox = gr.Textbox(label="Input", placeholder="Enter anything to start chatting"), 
    submit_btn=None
)

combined_interface = gr.TabbedInterface(
    title="Dell Virtual Technologist",
    interface_list=[database_interface, model_selection_interface, attributes_interface, chat_interface],
    tab_names=["Database", "Model", "Attributes", "Chatbot"],
    theme=gr.themes.Soft(),
)

combined_interface.queue().launch()