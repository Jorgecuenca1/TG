import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import PeftModel

# Configurar fragmentación de memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Limpiar la memoria de la GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Directorio para guardar el modelo ajustado
output_dir = "./finetuned_open_llama_7b"

# Cargar el modelo y el tokenizador
model_name = "openlm-research/open_llama_7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurar el token de padding
tokenizer.pad_token = tokenizer.eos_token

# Cargar el dataset para fine-tuning
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text']), batched=True)

# Configurar el collator de datos
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configurar los argumentos de entrenamiento mínimos
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,  # Reducir a una sola época
    per_device_train_batch_size=1,  # Reducir el tamaño del batch al mínimo
    gradient_accumulation_steps=1,  # Acumular gradientes en un solo paso
    fp16=True,  # Usar mixed precision training
    save_steps=10_000,
    save_total_limit=1,
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo ajustado
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Cargar el modelo ajustado y el modelo original
finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
original_model = AutoModelForCausalLM.from_pretrained(model_name)

# Combinar los pesos del modelo ajustado con el modelo original usando PEFT
peft_model = PeftModel(model=original_model, fine_tuned_model=finetuned_model)

# Guardar el modelo combinado
combined_model_output_dir = "./combined_open_llama_7b"
peft_model.save_pretrained(combined_model_output_dir)
tokenizer.save_pretrained(combined_model_output_dir)

print("Finetuning y combinación completados. El modelo combinado se ha guardado en:", combined_model_output_dir)
