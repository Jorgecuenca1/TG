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

# Cargar un modelo LLaMA de 33B cuantizado para caber en ~47 GB GPU
model_name = "facebook/llama-33b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
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
    full = "\n".join(texts)
    return full[-2000:]

def generate_answer(prompt: str, temperature: float, num_beams: int, max_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            temperature=temperature,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return txt.split("Respuesta:")[-1].strip()

class Query(BaseModel):
    question: str

@app.post("/send_message/")
async def send_message(query: Query):
    text = query.question.strip().lower()

    # Manejo de casos fijos
    fixed = None
    if any(g in text for g in ["hola", "hi", "¿cómo estás", "como estas"]):
        fixed = "Hola, ¿cómo estás? Hablas con el Asistente de Bitlink. Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."
    elif "quién eres" in text or "con quién hablo" in text:
        fixed = "Hablas con el Asistente de Bitlink. Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."
    elif any(k in text for k in ["cotiza", "precio", "presupuesto", "estimado"]):
        fixed = "Actualmente no manejo cotizaciones automáticas, te paso con un humano para darte un estimado en COP. Contacta al +57 310 2337052 (Jorge Cuenca)."

    if fixed:
        # Devolver dos opciones idénticas para la UI de "Mejor respuesta"
        return {"answer1": fixed, "answer2": fixed}

    # Contexto RAG
    context = retrieve_text_from_pdf("bitlink.pdf")

    # Generar dos respuestas con diferentes parámetros
    prompt1 = prompt_template.substitute(context=context, question=query.question)
    prompt2 = prompt_template.substitute(context=context, question=query.question)

    answer1 = generate_answer(prompt1, temperature=0.7, num_beams=5, max_tokens=200)
    answer2 = generate_answer(prompt2, temperature=1.0, num_beams=3, max_tokens=200)

    # Asegurar contacto al final
    for ans in (answer1, answer2):
        if "+57" not in ans:
            ans += " Para más información contáctanos al +57 310 2337052 (Jorge Cuenca)."
    return {"answer1": answer1, "answer2": answer2}

# (Los demás endpoints /best_answer/, /get_best_answers/, /upload_pdf/ permanecen igual)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
