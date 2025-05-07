from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


class Utils:
    def __init__(self, embed_model):
        self.embed_model_name = embed_model

    def utils_load(self):
        local_db = FAISS.load_local(
            "/Users/omarelkhashab/PycharmProjects/Local_RAG/local_index/bge-small-en-v1.5",
            self.embed_model_name,
            allow_dangerous_deserialization=True,
        )

        retriever_pdf = local_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "lambda_mult": 0.4,
                "filter": {"source": "2024-conocophillips-proxy-statement"},
            },
        )

        # llm1 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct") # Optional
        llm = ChatGroq(model='llama-3.1-8b-instant')

        return retriever_pdf, llm
