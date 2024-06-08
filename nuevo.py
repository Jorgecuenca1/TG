from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fitz  # PyMuPDF

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo y el tokenizador de Hugging Face
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Mover el modelo a la CPU
device = torch.device("cpu")
model.to(device)

# Función para recuperar texto desde un PDF
def retrieve_text_from_pdf(file_path):
    document = fitz.open(file_path)
    texts = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        texts.append(page.get_text())
    return texts

# Modelo para las solicitudes de entrada
class Query(BaseModel):
    question: str

# Endpoint de consulta
@app.post("/query/")
async def query(query: Query):
    # Recuperar los documentos relevantes desde un PDF
    pdf_path = "Solar-System-Wikipedia.pdf"  # Reemplaza esto con la ruta a tu PDF
    retrieved_docs = retrieve_text_from_pdf(pdf_path)
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
