from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os


os.environ["TRANSFORMERS_NO_TF"] = "1"


# Khai báo biến
data_path = "data"
model_path = "models/all-MiniLM-L6-v2"
vector_db_path = "vectorstores/db_faiss"


# Tạo ra vector db từ 1 đoạn text
def create_db_from_text():
    raw_text = """Điều 2. Cơ sở của trách nhiệm hình sự
                    1. Chỉ người nào phạm một tội đã được Bộ luật Hình sự quy định mới phải chịu trách nhiệm hình sự.
                    2. Chỉ pháp nhân thương mại nào phạm một tội đã được quy định tại Điều 76 của Bộ luật này mới phải chịu trách nhiệm hình sự.
                    Điều 3. Nguyên tắc xử lý
                    1. Đối với người phạm tội:
                    a) Mọi hành vi phạm tội do người thực hiện phải được phát hiện kịp thời, xử lý nhanh chóng, công minh theo đúng pháp luật;
                    b) Mọi người phạm tội đều bình đẳng trước pháp luật, không phân biệt giới tính, dân tộc, tín ngưỡng, tôn giáo, thành phần, địa vị xã hội;
                    c) Nghiêm trị người chủ mưu, cầm đầu, chỉ huy, ngoan cố chống đối, côn đồ, tái phạm nguy hiểm, lợi dụng chức vụ, quyền hạn để phạm tội; 
                    d)[3] Nghiêm trị người phạm tội dùng thủ đoạn xảo quyệt, có tổ chức, có tính chất chuyên nghiệp, cố ý gây hậu quả đặc biệt nghiêm trọng.
                    Khoan hồng đối với người tự thú, đầu thú, thành khẩn khai báo, tố giác đồng phạm, lập công chuộc tội, tích cực hợp tác với cơ quan có trách nhiệm trong việc phát hiện tội phạm hoặc trong quá trình giải quyết vụ án, ăn năn hối cải, tự nguyện sửa chữa hoặc bồi thường thiệt hại gây ra;
                    đ) Đối với người lần đầu phạm tội ít nghiêm trọng, thì có thể áp dụng hình phạt nhẹ hơn hình phạt tù, giao họ cho cơ quan, tổ chức hoặc gia đình giám sát, giáo dục;
                    e) Đối với người bị phạt tù thì buộc họ phải chấp hành hình phạt tại các cơ sở giam giữ, phải lao động, học tập để trở thành người có ích cho xã hội; nếu họ có đủ điều kiện do Bộ luật này quy định, thì có thể được xét giảm thời hạn chấp hành hình phạt, tha tù trước thời hạn có điều kiện;
                    g) Người đã chấp hành xong hình phạt được tạo điều kiện làm ăn, sinh sống lương thiện, hòa nhập với cộng đồng, khi có đủ điều kiện do luật định thì được xóa án tích.
                    2. Đối với pháp nhân thương mại phạm tội:
                    a) Mọi hành vi phạm tội do pháp nhân thương mại thực hiện phải được phát hiện kịp thời, xử lý nhanh chóng, công minh theo đúng pháp luật;
                    b) Mọi pháp nhân thương mại phạm tội đều bình đẳng trước pháp luật, không phân biệt hình thức sở hữu và thành phần kinh tế;
                    c) Nghiêm trị pháp nhân thương mại phạm tội dùng thủ đoạn tinh vi, có tính chất chuyên nghiệp, cố ý gây hậu quả đặc biệt nghiêm trọng;
                    d)[4] Khoan hồng đối với pháp nhân thương mại tích cực hợp tác với cơ quan có trách nhiệm trong việc phát hiện tội phạm hoặc trong quá trình giải quyết vụ án, tự nguyện sửa chữa, bồi thường thiệt hại gây ra, chủ động ngăn chặn hoặc khắc phục hậu quả xảy ra.
                    """

    # Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
        )
    
    chunks = text_splitter.split_text(raw_text)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Đưa vào Faiss Vector BD
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Vector Database saved to ./vectorstores/db_faiss")

    return db


# Tạo ra vector db từ file data
def create_db_from_files():
    # Khai data_loader quet qua toan bo tep trong data
    data_loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = data_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    print("Vector Database saved to ./vectorstores/db_faiss")

    return db


create_db_from_files()