from pathlib import Path
from app.pdf_loader import PDFLoader
from app.text_preprocessor import TextPreprocessor
from app.document_assembler import DocumentAssembler
from app.chunker import LegalSemanticChunker
from app.embeddings import EmbeddingGenerator
from app.vector_store import VectorStore
import shutil


from fastapi import APIRouter, File, HTTPException, UploadFile, status

router = APIRouter(tags=["Upload"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PDFLoader()
        preprocessor = TextPreprocessor()
        assembler = DocumentAssembler()
        chunker = LegalSemanticChunker()
        embedder = EmbeddingGenerator()
        store = VectorStore()

        # Load PDF
        pages = loader.load_pdf(str(file_path))

        # Clean pages
        pages = preprocessor.preprocess_pages(pages)

        # Assemble
        documents = assembler.assemble_documents(pages)

        # Chunk
        chunks = chunker.chunk_documents(documents)

        # Generate embeddings
        texts = [chunk["chunk_text"] for chunk in chunks]

        vectors = embedder.embed_batch(texts)

        # Store in Pinecone
        store.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
        )

        return {
            "status": "success",
            "filename": file.filename,
            "document_name": file.filename,
            "chunks": len(chunks),
        }
    except Exception as e:
        # remove uploaded file on error
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))