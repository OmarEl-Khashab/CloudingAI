from RAG.utils import Utils
from RAG.graph import RAGProcessor
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os


class RAGPipeline:
    def __init__(self, embeddings_model="BAAI/bge-small-en-v1.5",
                 ):
        self.embeddings_model = embeddings_model

    def setup_generator(self):

        key = os.environ["HUGGINGFACEHUB_API_TOKEN"]
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=key,
            model_name="BAAI/bge-small-en-v1.5",
            api_url="https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
        )

        utils = Utils(embed_model=embeddings)
        retriever, llm = utils.utils_load()
        template_rag = """You are an assistant for question-answering tasks. 

                Use the below retrieved context to answer USER question:

                {context} 

                Think carefully about the above context.

                Now, review the user question:

                {question}

                Provide an answer to this questions using only the above context or say it's beyond my knowledge.

                Use 10 lines maximum and keep the answer concise.

                Answer:"""

        processor = RAGProcessor(retriever, llm, template_rag)
        graph = processor.graph_build(processor)

        return graph

    def run(self, query: str) -> str:
        graph = self.setup_generator()

        initial_state = {
            "question": query
        }

        try:
            results = graph.invoke(initial_state)
            return results['messages'][0].content
        except Exception as e:
            print(f"Error during pipeline run: {e}")
            return "Sorry, I couldn't process your request."
