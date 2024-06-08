from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import locale
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo GPT-2 y el tokenizador
model_name = "TheBloke/Llama-2-13b-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
gen_cfg = GenerationConfig.from_pretrained(model_name)
gen_cfg.max_new_tokens = 512
gen_cfg.temperature = 0.0000001
gen_cfg.return_full_text = True
gen_cfg.do_sample = True
gen_cfg.repetition_penalty = 1.11

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=gen_cfg
)

llm = HuggingFacePipeline(pipeline=pipe)

# Test LLM with Llama 2 prompt structure and LangChain PromptTemplate


locale.getpreferredencoding = lambda: "UTF-8"

# RAG from web pages
web_loader = UnstructuredURLLoader(urls=["https://en.wikipedia.org/wiki/Solar_System"], mode="elements", strategy="fast")
web_doc = web_loader.load()
updated_web_doc = filter_complex_metadata(web_doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunked_web_doc = text_splitter.split_documents(updated_web_doc)
print(f"Number of chunks: {len(chunked_web_doc)}")

# Create vector database with FAISS
embeddings = HuggingFaceEmbeddings()
db_web = FAISS.from_documents(chunked_web_doc, embeddings)

# Use RetrievalQA chain
prompt_template = """
<s>[INST] <<SYS>>
Use the following context to Answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question. Don't try to make up an answer.

<</SYS>>

{context}

Question: {question} [/INST]
"""

@app.post("/send_message")
async def send_message(message: dict):
    try:
        user_message = message.get("message", "")

        # Tokenizar el mensaje del usuario
        input_ids = tokenizer.encode(user_message, return_tensors="pt")

        # Generar las dos respuestas únicas del modelo
        responses = set()
        while len(responses) < 2:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=40 + len(responses) * 10,  # Incrementar la longitud máxima
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

        return {"messages": list(responses)}
    except Exception as e:
        return {"error": str(e)}


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