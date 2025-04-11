"""
En este script se mostrará como generar un resumen de los textos 
Uno será con un modelo pre entrenado (T5 de Google)
El otro será utilizar Gemini para poder generar los resúmenes (Proceso lento)
"""
#### Iris Startup Lab
#### Fernando Dorantes Nieto

import os 
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re 
import google.generativeai as genai
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import time 

#### Modelos recomendados 
'models/gemma-3-27b-it'
'models/gemini-2.0-flash'

#### Para ver los modelos
'''
import pprint
for model in genai.list_models():
    pprint.pprint(model)
'''



genai.configure(api_key='la api key que se obtiene en AI Studio')

tokenizer = T5Tokenizer.from_pretrained("t5-base")
modelSummary = T5ForConditionalGeneration.from_pretrained("t5-base")


#### Función para generar el resumen del texto a través del transformer 
def generate_summary(text, max_length=150):
    input_text = "summarize: " + str(text)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = modelSummary.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#### Generar el resumen con Gemini 
def getSummaryIA(text):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    answer = model.generate_content(f"Resumir el siguiente texto: {text}")
    return answer.text

#### Debido a que la API de Gemini tiene uso de 10 solicitudes por minuto, se creó esta función que aplica estas funciones a todo
def processDfLimit(df, columna_texto, funcion_api, column_name):
    """
    Procesa un DataFrame aplicando una función de API a cada fila,
    respetando un límite de frecuencia de 10 solicitudes por minuto.
    """
    resultados = []
    solicitudes_minuto = 0
    inicio_minuto = time.time()

    for texto in df[columna_texto]:
        # Added try-except block to handle ReadTimeout errors
        try:
            resultado = funcion_api(texto)
        except  requests.exceptions.ReadTimeout:
            print(f"ReadTimeout error for text: {texto[:50]}...")
            resultado = None
        resultados.append(resultado)
        solicitudes_minuto += 1

        if solicitudes_minuto >= 10:
            tiempo_transcurrido = time.time() - inicio_minuto
            if tiempo_transcurrido < 60:
                time.sleep(60 - tiempo_transcurrido)
            solicitudes_minuto = 0
            inicio_minuto = time.time()

    df[column_name] = resultados
    return df