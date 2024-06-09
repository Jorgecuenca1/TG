from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fitz  # PyMuPDF
import os
import json

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo y el tokenizador de Hugging Face
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Asegúrate de mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
class Message(BaseModel):
    message: str

# Endpoint de consulta
@app.post("/send_message/")
async def send_message(message: Message):
    try:
        user_message = message.message

        # Recuperar los documentos relevantes desde un PDF
        pdf_path = "bitlink.pdf"  # Reemplaza esto con la ruta a tu PDF
        retrieved_docs = retrieve_text_from_pdf(pdf_path)
        retrieved_text = "\n".join(retrieved_docs)

        # Generar dos respuestas únicas del modelo
        responses = set()
        while len(responses) < 2:
            prompt = f"Pregunta: {user_message}\nDocumentos recuperados:\n{retrieved_text[:2048]}\nRespuesta:"
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=50 + len(responses) * 10,  # Incrementar la longitud máxima
                    num_beams=3 + 2 * len(responses),  # Variar el número de beams
                    no_repeat_ngram_size=2,
                    temperature=0.8 + 0.2 * len(responses)  # Ajustar la temperatura
                )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Añadir la respuesta al conjunto si no es una repetición
            responses.add(response)

        # Convertir el conjunto a una lista para retornar como respuesta
        return {"messages": list(responses)}
    except Exception as e:
        return {"error": str(e)}

# Endpoint para guardar la mejor respuesta
@app.post("/best_answer")
async def best_answer(answer: dict):
    try:
        user_message = answer.get("question", "")
        best_response = answer.get("response", "")

        # Ruta al archivo JSON donde se guardarán las preguntas y respuestas
        filename = "best_answers.json"

        # Leer el archivo existente o crear uno nuevo si no existe
        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
        else:
            data = []

        # Agregar la nueva pareja pregunta-respuesta
        data.append({user_message: best_response})

        # Escribir los datos actualizados de nuevo al archivo JSON
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
