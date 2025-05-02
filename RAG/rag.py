from RAG.utils import Utils
from RAG.graph import RAGProcessor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class RAGPipeline:
    def __init__(self, embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
                 ):
        self.embeddings_model = embeddings_model

    def setup_generator(self):
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={
                'device': 'cpu'
            },
        )

        utils = Utils(embed_model=embedding_model)
        retriever, llm = utils.utils_load()
        template_rag = """You are an assistant for question-answering tasks. 

                Use the below pieces of retrieved context to answer the question:

                {context} 

                Think carefully about the above context.

                Now, review the user question:

                {question}

                Provide an answer to this questions using only the above context or say it's beyond my knowledge. 

                Use 7 lines maximum and keep the answer concise.

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

