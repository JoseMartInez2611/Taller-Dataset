import exercise
import pandas as pd
import numpy as np
from datetime import datetime

#Ejercicio #1:
def test_first_point():
    data = {
        "edad": [22, 27, np.nan, 30, 120, 28, 27],
        "ingreso": [1200, 1500, 2000, np.nan, 9999999, 1800, 1500],
        "genero": ["M", "F", "F", None, "M", "F", "F"],
        "comprar": ["Si", "No", "Si", "No", "Si", "No", "No"]
    }

    df = pd.DataFrame(data)

    print("\nDataFrame original:\n" + str(df) + "\n")

    df = exercise.imput_values(df)
    df = exercise.delete_outliers(df)
    df = exercise.delete_duplicates(df)
    df = exercise.convert_binary_buy(df)
    df, scaler = exercise.scalar_variables(df)
    X, y = exercise.separate_xy(df)

    print(
        "Test Primer Punto:\n"
        "DataFrame procesado:\n" + str(df) + "\n"
        "Variables X:\n" + str(X) + "\n"
        "Variable y:\n" + str(y) + "\n"
    )



def ejercicio2():
    # Dataset
    data = {
        "nombre": ["Ana Lopez", "Juan Perez", "Maria Gomez"],
        "fecha_nacimiento": ["1998-05-10", "1990-08-15", "2000-01-20"],
        "ingreso": [1500, 2500, 1800]
    }

    df = pd.DataFrame(data)
    df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'])

    current_year = datetime.now().year
    df['edad'] = current_year - df['fecha_nacimiento'].dt.year

    nombres_separados = df['nombre'].str.split(' ', expand=True)
    df['primer_nombre'] = nombres_separados[0]
    df['apellido'] = nombres_separados[1]
    def clasificar_ingreso(ingreso):
        if ingreso < 1500:
            return 'Bajo'
        elif 1500 <= ingreso <= 2500:
            return 'Medio'
        else:
            return 'Alto'

    df['tipo_ingreso'] = df['ingreso'].apply(clasificar_ingreso)

    df_result = df.drop(columns=['nombre', 'fecha_nacimiento']).rename(columns={'primer_nombre': 'nombre'})

    print(df_result)
    
    return df_result

def ejercicio_3():
    data = {
        "ciudad":["bogota", "Bogotá", "BOGOTA", "medellin", "Medellín"],
        "genero": ["M", "Masculino", "F", "Femenino", "f"],
        "edad":[25, 30, 22, 28, 35]
    }   

    df = pd.DataFrame(data)
    exercise.normalizar_texto(df)
    exercise.unificar_categorías(df)
    exercise.validar_categorías_duplicadas(df)
    exercise.mostrar_valores_únicos(df)


def main():
    print("Ejercicio #1")
    test_first_point()
    print("Ejercicio #2")
    ejercicio2()
    print("Ejercicio #3")
    ejercicio_3()


main()

