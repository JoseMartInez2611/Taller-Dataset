import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Ejercicio #1

data={
    "edad":[22, 27, pn.nan, 30, 120, 28, 27],
    "ingreso":[1200, 1500, 2000, np.nan, 9999999, 1800, 1500],
    "genero":["M", "F", "F", None, "M", "F", "F"],
    "comprar":["Si", "No", "Si", "No", "Si", "No", "No"]
}

Detectar valores faltantes
● Imputar:
    ○ edad → mediana
    ○ ingreso → media
    ○ genero → moda
● Detectar y eliminar outliers extremos (edad > 100, ingreso muy alto)
● Eliminar duplicados
● Convertir compra a binaria
● Escalar edad e ingreso
● Separar X e y

"""

# 1. Imputar valores faltantes
def imput_values(df):
    df = df.copy()
    
    # edad → mediana
    df["edad"] = df["edad"].fillna(df["edad"].median())
    
    # ingreso → media
    df["ingreso"] = df["ingreso"].fillna(df["ingreso"].mean())
    
    # genero → moda
    df["genero"] = df["genero"].fillna(df["genero"].mode()[0])
    
    return df


# 2. Detectar y eliminar outliers extremos
def delete_outliers(df):
    df = df.copy()
    
    # edad > 100
    df = df[df["edad"] <= 100]
    
    # ingreso muy alto (ejemplo: percentil 99)
    limite_ingreso = df["ingreso"].quantile(0.99)
    df = df[df["ingreso"] <= limite_ingreso]
    
    return df


# 3. Eliminar duplicados
def delete_duplicates(df):
    return df.drop_duplicates().copy()


# 4. Convertir compra a binaria
def convert_binary_buy(df):
    df = df.copy()
    
    df["comprar"] = df["comprar"].map({
        "Si": 1,
        "No": 0
    })
    
    return df


# 5. Escalar edad e ingreso
def scalar_variables(df):
    df = df.copy()
    
    scaler = StandardScaler()
    df[["edad", "ingreso"]] = scaler.fit_transform(df[["edad", "ingreso"]])
    
    return df, scaler


# 6. Separar X e y
def separate_xy(df):
    X = df.drop("comprar", axis=1)
    y = df["comprar"]
    
    return X, y



"""
Ejercicio #2

data={
    "nombre":["Ana Lopez", "Juan Perez", "Maria Gomez"],
    "fecha_nacimiento":["1998-05-10", "1990-08-15", "2000-01-20"],
    "ingreso":[1500, 2500, 1800]

}

● Convertir fecha_nacimiento a tipo fecha
● Crear nueva columna edad
● Separar:
    ○ nombre → nombre, apellido
● Eliminar columnas innecesarias
● Clasificar ingreso en:
    ○ bajo (<1500)
    ○ medio (1500–2500)
    ○ alto (>2500)


"""



"""
Ejercicio #3

data{
    "ciudad":["bogota", "Bogotá", "BOGOTA", "medellin", "Medellín"],
    "genero:"["M", "Masculino", "F", "Femenino", "f"],
    "edad":[25, 30, 22, 28, 35]
}

● Normalizar texto (minúsculas, quitar tildes )
● Unificar categorías:
    ○ bogota → Bogotá
    ○ masculino/M → M
    ○ femenino/F → F
● Validar que no existan categorías duplicadas
● Mostrar valores únicos finales

"""




"""
Ejercio #4

data = {
    "edad": [20, 25, None, 30, 35, 40, None],
    "salario": [1000, 2000, 3000, None, 5000, 6000, 7000],
    "departamento": ["ventas", "ventas", "IT", "IT", None, "HR", "HR"],
    "rendimiento": ["alto", "medio", "alto", "bajo", "medio", "alto", "bajo"]
}

● Imputar valores faltantes correctamente según tipo
● Codificar:
    ○ rendimiento (ordinal: bajo < medio < alto)
● One-hot encoding en departamento
● Normalizar salario


"""


