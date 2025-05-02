from fastapi import APIRouter, HTTPException
from App.api.models.validators import QueryRequest, QueryResponse
from RAG.rag import RAGPipeline

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def get_response(request: QueryRequest):
    try:
        print(f"Received request query: {request.query}")
        rag = RAGPipeline()
        output = rag.run(query=request.query)
        print(f"Sent Response of query: {output}")
        return QueryResponse(answer=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
