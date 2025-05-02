from fastapi import FastAPI, Request
from App.api.query import router as router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# Initialize FastAPI application
app = FastAPI(
    title="Clouding AI Local RAG",
    description="FastAPI for POC for Local RAG datasets using LangGraph and Ollama",
)

# CORS middleware for the frontend only to make work with any source
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Initialize templates and static files
templates = Jinja2Templates(directory="Frontend/Template")
app.mount("/static", StaticFiles(directory="Frontend/Static"), name="static")


# Serve the index.html page with the form
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "API is Up and Running"}


# Include the query and health endpoints
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
