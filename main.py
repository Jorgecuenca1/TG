from fastapi import FastAPI, File, UploadFile
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

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo y el tokenizador
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt con personalidad y servicios de Bitlink
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

def retrieve_text_from_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    doc = fitz.open(file_path)
    texts = [doc.load_page(i).get_text() for i in range(len(doc))]
    # Concatenar y truncar para ajustarse al prompt
    full = "\n".join(texts)
    return full[-2000:]

def generate_answer(prompt: str,
                    temperature: float = 0.7,
                    max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo después de la etiqueta “Respuesta:”
    return text.split("Respuesta:")[-1].strip()

class Query(BaseModel):
    question: str

@app.post("/send_message/")
async def send_message(query: Query):
    text = query.question.strip().lower()

    # Manejo de saludos
    if any(greet in text for greet in ["hola", "hi", "¿cómo estás", "como estas"]):
        return {"response": (
            "Hola, ¿cómo estás? Hablas con el Asistente de Bitlink. "
            "Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."
        )}

    # Manejo de identidad
    if "quién eres" in text or "con quién hablo" in text:
        return {"response": (
            "Hablas con el Asistente de Bitlink. "
            "Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."
        )}

    # Manejo de cotización
    if any(keyword in text for keyword in ["cotiza", "precio", "presupuesto", "estimado"]):
        return {"response": (
            "Actualmente no manejo cotizaciones automáticas, te paso con un humano "
            "para darte un estimado en COP. "
            "Contacta a +57 310 2337052 (Jorge Cuenca)."
        )}

    # Si no era ninguno de los anteriores, usa RAG + LLM
    context = retrieve_text_from_pdf("bitlink.pdf")
    prompt = prompt_template.substitute(context=context, question=query.question)

    answer = generate_answer(prompt)
    # Asegúrate de incluir el contacto
    if "+57" not in answer:
        answer += " Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."

    return {"response": answer}


@app.post("/best_answer/")
async def best_answer(answer: BestAnswer):
    try:
        # Guardar en JSON
        fname = "best_answers.json"
        data = json.load(open(fname)) if os.path.exists(fname) else []
        data.append({answer.question: answer.response})
        json.dump(data, open(fname, "w"), indent=4)

        # Añadir al PDF
        doc = fitz.open("bitlink.pdf")
        page = doc.load_page(-1)
        text = f"Question: {answer.question}\nAnswer: {answer.response}"
        rect = fitz.Rect(72, 72, 500, 200)
        page.insert_textbox(rect, text, fontsize=12)
        doc.save("bitlink.pdf")

        return {"status": "success"}
    except Exception as e:
        logging.error("Error in best_answer:", exc_info=e)
        return {"status": "error", "message": str(e)}

@app.get("/get_best_answers/")
async def get_best_answers():
    try:
        fname = "best_answers.json"
        if not os.path.exists(fname):
            return JSONResponse({"message": "File not found"}, status_code=404)
        return JSONResponse(open(fname).read())
    except Exception as e:
        logging.error("Error in get_best_answers:", exc_info=e)
        return {"status": "error", "message": str(e)}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        upload_dir = "static"
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logging.error("Error in upload_pdf:", exc_info=e)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
