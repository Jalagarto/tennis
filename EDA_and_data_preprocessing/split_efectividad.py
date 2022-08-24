# entrenar un modelo para cada tipo de efectividad, pero no para E>4

import pandas as pd

def load_data(filepath):
    df = pd.read_excel(f, engine='openpyxl')
    return df


def main(f):
    return load_data(f)

if __name__=="__main__":
    f = '/home/javier/mis_proyectos/DATAJAVI_V3_noblank.xlsx'  # clean file (one column was wrong --> 10e5 rows)
    df = main(f)
    print(df.head())