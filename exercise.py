import pandas as pd
from datetime import datetime

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
