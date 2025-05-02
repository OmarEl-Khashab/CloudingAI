from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_cloud_services import LlamaParse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class DataLoad:
    def __init__(self, datapath, embed_model_name, file_parser):

        self.datapath = datapath
        self.embed_model_name = embed_model_name
        self.file_parser = file_parser

    def data_load(self):
        # set up parser
        # Initialize the parser
        # "markdown" or "text"

        # Define the path to the single PDF file

        # Set up the file extractor for PDF
        file_extractor_pdf = {".pdf": parser}
        pdf_docs = []
        # Use SimpleDirectoryReader to parse the single file
        loader = [SimpleDirectoryReader(input_files=[self.datapath], file_extractor=file_extractor_pdf)]
        print("---DATA PARSED SUCCESSFULLY--")
        # Load the data
        for loader in loader:
            pdf_docs.append(loader.load_data())

        # Clean up metadata (optional)
        langchain_pdf_docs = []
        for doc_list in pdf_docs:  # Iterate directly over the outer list
            for doc in doc_list:  # Iterate directly over the inner list
                # Update metadata and create the new Document in one step
                updated_metadata = {"source": doc.metadata["file_name"].split(".")[0]}
                updated_content = " ".join(doc.text.split())
                langchain_pdf_docs.append(
                    Document(page_content=updated_content, metadata=updated_metadata)
                )

        text_split = RecursiveCharacterTextSplitter()

        pages_splits_pdf = text_split.split_documents(langchain_pdf_docs)
        print("---PAGES SPLITTER SUCCESSFUL--")

        return pages_splits_pdf

    def index_vector_store(self):

        pages_splits_pdf = self.data_load()

        model_kwargs = {'device': 'cpu', }
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs=model_kwargs
        )

        db = FAISS.from_documents(pages_splits_pdf, embedding_model)
        db.save_local("Local_Vector-Store_test_1")

        print("---VECTOR STORE INDEX CREATED--")




file_path_pdf = "/data/2024-conocophillips-proxy-statement.pdf"
model_name = 'all-mpnet-base-v2'

parser = LlamaParse(result_type="markdown")

x = DataLoad(datapath=file_path_pdf, embed_model_name=model_name, file_parser=parser)
x.index_vector_store()
