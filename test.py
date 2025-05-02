import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AnyMessage
from RAG.utils import Utils
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class State(TypedDict):
    question: str
    generation: str
    messages: Annotated[list[AnyMessage], operator.add]

class RAGProcessor:
    def __init__(self, retriever, llm, template_rag: str):
        self.retriever = retriever
        self.llm = llm
        self.template_rag = template_rag

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate(self, state: State) -> dict:
        print("---GENERATION---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        formatted_doc = self.format_docs(documents)
        prompt = self.template_rag.format(context=formatted_doc, question=question)
        response = self.llm.invoke([HumanMessage(prompt)])
        return {
            "generation": response.content,
            "messages": [response],
        }


model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {
    'device': 'cpu'
}

embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
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

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("llm", processor.generate)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)
graph = graph_builder.compile()

initial_state_1 = {"question": "Who are the 12 director nominees that board recommends?"}
results = graph.invoke(initial_state_1)

print(results['messages'][0].content)
