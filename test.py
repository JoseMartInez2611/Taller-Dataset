import exercise
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def main():
    test_first_point();



main()
