from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA


model_path = "models/vinallama-7b-chat_q5_0.gguf"
model_embedding_path = "models/all-MiniLM-L6-v2"
vector_db_path = "vectorstores/db_faiss"


# Load models
def load_model(model_path: str):
    model = CTransformers(
        model= model_path,
        model_type= "llama",
        max_new_tokens= 1024,
        temperature= 0.01,
    )
    return model

# Tao promt template
def create_prompt(template: str):
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
    return prompt
# 
def create_qa_chain(prompt, model, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm= model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k":3}),
        return_source_documents = True,
        chain_type_kwargs={"prompt":prompt}
    )
    return llm_chain

# Read from VectorDB
def read_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_embedding_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True
        )
    return db


if __name__ == "__main__":
    template = """<|im_start|>system
                Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói "tôi không biết", đừng cố tạo ra câu trả lời nhiễu.
                {context}<|im_end|>
                <|im_start|>user
                {question}<|im_end|>
                <|im_start|>assistant
                """
    prompt = create_prompt(template)

    db = read_vector_db()
    llm = load_model(model_path)
    llm_chain = create_qa_chain(prompt, llm, db)

    question = "Đối tượng áp dụng của luật đất đai?"
    result = llm_chain.invoke({"query": question})
    answer = result.get("result") or result.get("output_text") or result
    
    print(answer)

    if isinstance(result, dict) and "source_documents" in result:
        print("\nNGUỒN THAM CHIẾU")
        for i, doc in enumerate(result["source_documents"], 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            source = meta.get("source") or meta.get("file_path") or meta
            print(f"[{i}] {source}")