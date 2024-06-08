from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fitz  # PyMuPDF
from langchain.chains import RetrievalAugmentedGeneration
from langchain.retrievers import PdfDocumentRetriever

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo y el tokenizador de Hugging Face
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Asegúrate de mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configurar LangChain con PyMuPDF para manejar PDFs
class PDFRetriever(PdfDocumentRetriever):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.document = fitz.open(file_path)

    def get_documents(self):
        texts = []
        for page_num in range(len(self.document)):
            page = self.document.load_page(page_num)
            texts.append(page.get_text())
        return texts

# Configurar la cadena de RAG
pdf_path = "Solar-System-Wikipedia.pdf"  # Reemplaza esto con la ruta a tu PDF
retriever = PDFRetriever(pdf_path)
rag_chain = RetrievalAugmentedGeneration(retriever=retriever)

# Modelo para las solicitudes de entrada
class Query(BaseModel):
    question: str

# Endpoint de consulta
@app.post("/query/")
async def query(query: Query):
    # Recuperar los documentos relevantes
    retrieved_docs = retriever.get_documents()
    retrieved_text = "\n".join(retrieved_docs)

    # Generar una respuesta usando el modelo de lenguaje
    prompt = f"Pregunta: {query.question}\nDocumentos recuperados:\n{retrieved_text}\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
