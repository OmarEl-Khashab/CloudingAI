from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

class Utils:
    def __init__(self, embed_model):
        self.embed_model_name = embed_model

    def utils_load(self):
        local_db = FAISS.load_local(
            "/Users/omarelkhashab/PycharmProjects/Local_RAG/local_index/Local_Vector-Store-all-MiniLM-L6-v2",
            self.embed_model_name,
            allow_dangerous_deserialization=True,
        )

        retriever_pdf = local_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "lambda_mult": 0.3,
                "filter": {"source": "2024-conocophillips-proxy-statement"},
            },
        )

        llm = ChatOllama(model="mistral:latest", temperature=0)

        return retriever_pdf, llm
