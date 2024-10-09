# QA_chatbot_with_RAG

## Project Overview
#### This project about create an Vietnamese question-answer bot based on langchain and RAG, you can upload your own document and ask questions about the document.

## How to run this project
#### 1. Clone this repository to your local machine.
#### 2. Create models folder contains model file: [vinallama-7b-chat_q5_0.gguf](https://huggingface.co/RichardErkhov/vilm_-_vinallama-7b-chat-gguf/tree/main) or any other LLM model for Vietnamese.
#### 3. Create data folder contains your own document file (only pdf).
#### 4. Run prepare_vector_db.py for creating vector database for your document.
#### 5. Run qabot.py for enter your question and get the answer from your document file.

### **Note**: This project use Ctransformer - a type of quantize model, support for running faster on cpu but reduce the quality of the answer.