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
        print("---GENERATION ON PROCESS---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        formatted_doc = self.format_docs(documents)
        prompt = self.template_rag.format(context=formatted_doc, question=question)
        response = self.llm.invoke([HumanMessage(prompt)])
        return {
            "generation": response.content,
            "messages": [response],
        }

    def graph_build(self, processor):
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("llm", processor.generate)
        graph_builder.add_edge(START, "llm")
        graph_builder.add_edge("llm", END)
        graph = graph_builder.compile()
        return graph
