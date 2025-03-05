
"""
Usaremos una librería llamada GPT-2 y DeepESP (un modelo preentrenado de OpenAI)
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""
GPT2LMHeadModel que es el modelo de lenguaje de GPT-2 para generar texto,
GPT2Tokenizer, que se encarga de convertir texto en tokens (OneNote aportado: IA Generativa).
"""

#model_name = "gpt2"
model_name = "DeepESP/gpt2-spanish"  #Entiende mejor el español que el gpt2
model = GPT2LMHeadModel.from_pretrained(model_name) #From_pretrained: Descarga y carga tanto el modelo como el tokenizador del repositorio Hugging Face, permitiendo usarlos directamente sin necesidad de entrenamiento.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

"""
En gpt2 no hay un "pad token: token especial que se usa para completar entradas más cortas".
Se asigna el pad_token al token eos_token (fin de secuencia). Esto también evita advertencias relacionadas con entradas de diferentes longitudes
"""
tokenizer.pad_token = tokenizer.eos_token

# Texto base
initial_text = "Eramos Raúl y Yo contra el mundo"

"""
El texto inicial se convierte en tokens, retorna un tensor de PyTorch.
"""
inputs = tokenizer.encode(initial_text, return_tensors="pt")

"""
Hiperparamétros:
max_length=500: Longitud máxima del texto generado.
num_return_sequences=1: Número de secuencias a generar.
do_sample=True: Activa el muestreo aleatorio para que el texto sea más creativo.
temperature=0.7: Controla la "creatividad". Un valor más bajo genera texto más conservador; un valor más alto genera texto más diverso.
top_k=50: Limita las predicciones a las 50 palabras más probables, reduciendo las opciones obvias.
top_p=0.9: Filtrado basado en probabilidad acumulativa; se descartan palabras con probabilidad muy baja.
repetition_penalty=1.2: Penaliza repeticiones excesivas de palabras o frases.
no_repeat_ngram_size=2: Evita la repetición de frases o n-grams (en este caso, secuencias de 2 palabras).
pad_token_id=tokenizer.eos_token_id: Asigna el token de fin de secuencia como token de padding.

"""
outputs = model.generate(
    inputs,
    max_length=500,
    num_return_sequences=1,
    do_sample=True,  
    temperature=0.7,  
    top_k=50,  
    top_p=0.9,  
    repetition_penalty=1.2,  
    no_repeat_ngram_size=2,  
    pad_token_id=tokenizer.eos_token_id
)

"""
Decodifica los tokens, genera un texto legible. 
"""

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
