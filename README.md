# CloudingAI Local RAG Tool

##  Description
CloudingAI Local RAG Tool is a tool using Retrieval-Augmented Generation (RAG).
It is a web application that utilizes Open Source LLMs models using Groq API for On-Premises POC
to process and analyze user queries for internal data for CloudingAI
The tool provides valuable insights to CloudingAI clients by 
leveraging the capabilities of RAG and LangGraph.

The project comprises the following components::

- **Backend:** The question-and-answer chat feature is developed using Open Source with 'llama-3.1-8b-instant' model and LangGraph to implement Retrieval-Augmented Generation (RAG) 
for extracting information for internal PDF with Llama index to Parse complex documents, The backend connection is facilitated by a FastAPI-based service that processes user queries and interactions.

- **Self-Hosted Vector Store:** HuggingFace Embeddings models such as "BAAI/bge-small-en-v1.5"  using HuggingFaceInferenceAPIEmbeddings
- **PDF with 100k+ words and complex tables and graphs:** 2024-conocophillips-proxy-statement.pdf

- **Frontend:** A simple HTML/CSS interface that allows for an interactive user experience.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Instructions](#Instructions)
- [API Documentation](#api-documentation)
- [Acknowledgement](#Acknowledgement)

## Tool Used

- **LLM-Generated Results:** Utilizes Groq’s OpenSource LLM models API through retrieval-augmented generation (RAG) using LangGraph for accurate answers based on documents data given.
- **Self-Hosted Vector Store:** HuggingFaceInferenceAPIEmbeddings for embeddings models API.
- **Simple UI:** User-friendly question-answer interface designed for seamless interaction, enhancing user experience and engagement.

## Prerequisites

- **Python:** Required programming language; ensure version 3.9 or later is installed.
- **LangGraph:** Library for building LLM applications.
- **LlamaIndex:** Library for building LLM applications with Advanced Parsing Options.
- **FAISS DB:** Database for storing Embeddings Index
- **FastAPI:** Framework for building APIs.
- **Groq:** OpenSource Platform for LLM models APIs.
- **Hugging Face:** OpenSource Platform for LLM models and Embeddings.
- **API keys Generation:**   Official Websites for 'LLAMA_CLOUD_API_KEY', 'GROQ_API_KEY', 'HUGGINGFACEHUB_API_TOKEN' 
- **Conda Virtual Environment (optional):** To manage dependencies for running python packages.

## Installation and Setup

1. **Clone the Repository:**

	```sh
	git clone https://github.com/OmarEl-Khashab/CloudingAI.git
	cd Local_RAG
	```

2. **Install the required packages** 

	```sh
	pip install -r requirements.txt
	```
 
3. **Create Embedding Index :**

	Add your dataset path and save path of the embeddings store in the code:
	```sh
	python dataload.py
	```

4. **Start the App:**

	Run the main code to start the RAG experience in the UI:

	```sh
	python main.py
	```
 	Add your generated tokens and start the RAG experience 

## How to use:

1. **Access the Application:**

	- Open your browser and go to `http://127.0.0.1:8000`.


2. **Interact with the Application:**

	- Enter a question in the input field (e.g., “Who are the 12 director nominees that board recommends?”).
	- Click **Generate Results** to input the query.


3. **View Results:**
	- The LLM will generate a response the  displayed in the screen.
## API Documentation

The backend exposes a POST endpoint for querying datasets.

- **Endpoint:** `/query`
- **Method:** POST
- **Description:** Processes a user query and returns AI-generated Results.

**Request Body:**

```json
{
  "query": "Who are the 12 director nominees that the board recommends?"
}
```

**Response:**

```json
{
  "answer": "The 12 director nominees recommended by the board are Dennis V. Arriola, Ryan M. Lance, Timothy A. Leach, William H. McRaven, Robert A. Niblock, David T. Seaton, R.A. Walker, Arjun N. Murti, James J. Mulva, John W. Watson, Kenneth Frazier, and Mark S. Little."
}
```

**Error Handling:**

- If the request fails, the API returns an appropriate HTTP status code and an error message.

## Web Application Interface



<p align="center">
  <img src="/RAG.png" alt="My Image" width="700"/>
</p>

## Acknowledgement

This is a big thanks for Clouding AI for giving the opportunity to complete this Machine learning Challenge

Feel free to contribute reach out for the project by opening issues or submitting pull requests.
