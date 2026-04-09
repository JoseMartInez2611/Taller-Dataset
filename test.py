import pandas as pd
import numpy as np

from exercise import mostrar_valores_únicos, normalizar_texto, unificar_categorías, validar_categorías_duplicadas


def ejercicio_3():
    data = {
        "ciudad":["bogota", "Bogotá", "BOGOTA", "medellin", "Medellín"],
        "genero": ["M", "Masculino", "F", "Femenino", "f"],
        "edad":[25, 30, 22, 28, 35]
    }   
    
    df = pd.DataFrame(data)
    normalizar_texto(df)
    unificar_categorías(df)
    validar_categorías_duplicadas(df)
    mostrar_valores_únicos(df)


ejercicio_3()