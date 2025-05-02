# it allows you to define the structure of your request and response data,
# making sure that the data is valid and in the expected format.
from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str  # Define the input query to be passed to the pipeline


class QueryResponse(BaseModel):
    # answer: str  # Define the expected output after running the query
    answer: str
