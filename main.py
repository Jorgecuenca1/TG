 from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import fitz  # PyMuPDF
import os
import json
from string import Template
import logging
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
# Inicializar FastAPI
app = FastAPI()
# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes, cambiar a ["http://localhost:3000"] en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo y el tokenizador de Hugging Face
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#new model = AutoModelForCausalLM.from_pretrained("ruta/al/modelo2") # el nuevo model es el entrenamiento
#base_model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    low_cpu_mem_usage=True,
#    return_dict=True,
#    torch_dtype=torch.float16,
#    device_map = {"": [0, 1]}
#)

#model = PeftModel.from_pretrained(base_model, new_model)
#model = model.merge_and_unload()
# Recargar tokenizador para guardarlo
#tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
# Asegúrate de mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Template para el prompt
prompt_template = Template("""
<s>[INST] <<SYS>>
Eres el Asistente de Bitlink, especializado en los siguientes servicios de software:
- Desarrollo web y móvil
- Integración de APIs y RAG
- Despliegue en contenedores Docker
- Soporte y mantenimiento continuo

Reglas de interacción:
1. Si el usuario saluda (‘hola’, ‘hi’, ‘¿cómo estás?’), responde “Hola, ¿cómo estás? Hablas con el Asistente de Bitlink.”
2. Si pregunta “¿quién eres?” o “¿con quién hablo?”, responde “Hablas con el Asistente de Bitlink.”
3. Si pide cotización, responde “Actualmente no manejo cotizaciones automáticas, te paso con un humano para darte un estimado en COP.”
4. Al final de tu respuesta, incluye: “Para más información contáctanos al +57 310 2337052 (Jorge Cuenca).”
5. No inventes datos; si no sabes algo, ofrece pasar a un humano.

<</SYS>>

Contexto:
$context

Usuario: $question
Respuesta:
""")

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

# Función para generar respuesta con manejo de memoria
def generate_answer(prompt, temperature=0.8, num_beams=5, max_new_tokens=150, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            temperature=temperature,
            early_stopping=True,
            do_sample=do_sample
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la respuesta, asumiendo que la respuesta sigue después de 'Respuesta:'
    answer = answer.split("Respuesta:")[-1].strip()
    return answer

# Endpoint de consulta
@app.post("/send_message/")
async def send_message(query: Query):
    # Recuperar los documentos relevantes desde un PDF
    pdf_path = "bitlink.pdf"  # Reemplaza esto con la ruta a tu PDF
    retrieved_docs = retrieve_text_from_pdf(pdf_path)
    retrieved_text = "\n".join(retrieved_docs)

    # Usar el template para generar los prompts
    prompt1 = prompt_template.substitute(context=retrieved_text[:2048], question=query.question)
    prompt2 = prompt_template.substitute(context=retrieved_text[2048:4096], question=query.question)

    # Generar respuestas con parámetros diferentes para asegurar variedad
    answer1 = generate_answer(prompt1, temperature=0.8, num_beams=5, max_new_tokens=150, do_sample=True)
    answer2 = generate_answer(prompt2, temperature=1.0, num_beams=3, max_new_tokens=150, do_sample=True)

    return {"answer1": answer1, "answer2": answer2}

# Modelo para las solicitudes de entrada
class BestAnswer(BaseModel):
    question: str
    response: str

# Endpoint para guardar la mejor respuesta y agregarla al PDF
# Endpoint para guardar la mejor respuesta y agregarla al PDF
@app.post("/best_answer/")
async def best_answer(answer: BestAnswer):
    try:
        user_message = answer.question
        best_response = answer.response

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

        # Agregar la respuesta al PDF
        pdf_path = "bitlink.pdf"
        logging.info(f"Opening PDF: {pdf_path}")
        document = fitz.open(pdf_path)
        page = document.load_page(-1)  # Obtener la última página del PDF

        text = f"Question: {user_message}\nAnswer: {best_response}"
        logging.info(f"Inserting text into PDF: {text}")
        rect = fitz.Rect(72, 72, 500, 200)  # Ajustar las coordenadas y el tamaño del cuadro de texto
        page.insert_textbox(rect, text, fontsize=12, fontname="helv")
        document.save(pdf_path)
        logging.info(f"PDF saved: {pdf_path}")

        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"status": "error", "message": str(e)}
# Endpoint para obtener el contenido del archivo best_answers.json
@app.get("/get_best_answers/")
async def get_best_answers():
    try:
        filename = "best_answers.json"

        # Verificar si el archivo existe
        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
            return JSONResponse(content=data)
        else:
            return JSONResponse(content={"message": "File not found"}, status_code=404)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"status": "error", "message": str(e)}

# Endpoint para subir y guardar un archivo PDF
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Directorio donde se guardarán los archivos subidos
        upload_dir = "static"

        # Crear el directorio si no existe
        os.makedirs(upload_dir, exist_ok=True)

        # Ruta completa del archivo subido
        file_path = os.path.join(upload_dir, file.filename)

        # Guardar el archivo
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logging.error(f"Error occurred while uploading file: {str(e)}")
        return {"status": "error", "message": str(e)}
# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


