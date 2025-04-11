"""
En este script se mostrará como generar un resumen de los textos 
Uno será con un modelo pre entrenado (T5 de Google)
El otro será utilizar Gemini para poder generar los resúmenes (Proceso lento)
"""
#### Iris Startup Lab
#### Fernando Dorantes Nieto

import pandas as pd 
import numpy as np 
from deep_translator import GoogleTranslator


def anyLanguageToSpanish(text, language):
    text = str(text)
    return GoogleTranslator(source = language, target='es').translate(text = text)
#### Este usa tu cuenta de Google Implicita en tu navegador default

