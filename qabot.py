from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from warnings import filterwarnings

filterwarnings("ignore")

class QABot:
    def __init__(self, model_file, vector_db_path):
        self.model_file = model_file
        self.vector_db_path = vector_db_path
        self.llm = self.load_llm(model_file)
        self.db = self.read_vectors_db()
        self.prompt = self.create_prompt()

    # Hàm load LLM Model
    def load_llm(self, model_file):
        llm = CTransformers(
            model=model_file,
            model_type="llama",
            max_new_tokens=1024,
            temperature=0.01
        )
        return llm

    # Tạo prompt template
    def create_prompt(self):
        # Tạo Prompt
        template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
            {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return prompt

    # Tạo simple chain
    def create_qa_chain(self):
        llm_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
            return_source_documents=False,
            chain_type_kwargs={'prompt': self.prompt}
        )
        return llm_chain

    # Read từ VectorDB
    def read_vectors_db(self):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(self.vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db

    # Chạy chatbot
    def run_chatbot(self, question):
        llm_chain = self.create_qa_chain()
        response = llm_chain.invoke({"query": question})
        output = response.get('result')
        return output 

# Cấu hình đường dẫn model và vector DB
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

bot = QABot(model_file, vector_db_path)

user_input = input("Nhập câu hỏi: ")
bot_response = bot.run_chatbot(user_input)
output = bot_response.split('\n')
bot_response = output[0] if output[0] != "" else output[1]
print(bot_response)