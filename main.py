from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import os

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
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

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