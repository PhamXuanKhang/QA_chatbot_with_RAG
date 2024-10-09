import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from warnings import filterwarnings

filterwarnings("ignore")

# Tạo đường dẫn đến folder lưu trữ vector database
UPLOAD_DIRECTORY = "data"
vector_db_path = "vectorstores/db_faiss"

# Hàm tạo vector database từ các files trong folder
def create_db_from_files():
    # Quét toàn bộ thư mục data để lấy tất cả các PDF đã lưu
    loader = DirectoryLoader(UPLOAD_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Chia nhỏ tài liệu
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Tạo embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)

    # Lưu cơ sở dữ liệu vector
    db.save_local(vector_db_path)
    
    return db

create_db_from_files("a.pdf")