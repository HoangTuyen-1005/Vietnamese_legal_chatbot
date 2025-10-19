from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model_path = "models/vinallama-7b-chat_q5_0.gguf"

# Load models
def load_model(model_path):
    model = CTransformers(
        model = model_path,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01,
    )
    return model

# Tao promt template
def create_prompt(template):
    promt = ChatPromptTemplate.from_template(template)
    return promt

# Tao simple chain
def create_simple_chain(prompt, model):
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain


# Test chain
template = """<|im_start|>system
            Bạn là một trợ lí AI về luật Việt Nam. Hãy trả lời người dùng một cách chính xác.
            <|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
            """

promt = create_prompt(template)
llm = load_model(model_path)
llm_chain = create_simple_chain(promt, llm)

question = "retrieval augmented generation là gì?"
response = llm_chain.invoke({"question":question})
clean_response = response.split("<|im_end|>")[0].strip()
print(clean_response)