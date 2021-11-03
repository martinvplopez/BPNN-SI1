import numpy as np
import pandas as pd
import matplotlib.pyplot
import network

def test_0():
    #Convert excel to CSV to successful reading
    train = pd.read_excel('Datos_PrActica_1_BPNN.xls')
    train.to_csv('Datos_PrActica_1_BPNNenCSV.csv', index = False)
    df = pd.read_csv('Datos_PrActica_1_BPNNenCSV.csv')
    #Change the date and hour to just the hour and insert it on the first column of the frame
    df.insert(1, "HORA", df['FECHA_HORA'].str[10:16], True)
    df.drop('FECHA_HORA', inplace=True, axis=1)


    #Vectorice any string or object value to an integer to avoid any issues
    func = np.vectorize(conditions_tipo_precipitacion)
    tipo_precipitacion = func(df['TIPO_PRECIPITACION'])
    df.drop('TIPO_PRECIPITACION', inplace=True, axis=1)
    #df['TIPO_PRECIPITACION'] = tipo_precipitacion
    df.insert(8, "TIPO_PRECIPITACION",tipo_precipitacion, True)
    #print(df['TIPO_PRECIPITACION'])
    func1 = np.vectorize(conditions_intensidad)
    intensidad_precipitacion = func1(df['INTENSIDAD_PRECIPITACION'])
    df.drop('INTENSIDAD_PRECIPITACION', inplace=True, axis=1)
    df.insert(9, "INTENSIDAD_PRECIPITACION", intensidad_precipitacion, True)
    #print(df['INTENSIDAD_PRECIPITACION'])
    func2 = np.vectorize(conditions_estado_carretera)
    estado_carretera = func2(df['ESTADO_CARRETERA'])
    df.drop('ESTADO_CARRETERA', inplace=True, axis=1)
    df.insert(12, "ESTADO_CARRETERA", estado_carretera, True)
    #print(df['ESTADO_CARRETERA'])
    #Output
    y_data = df.iloc[0:200, 13].values
    print(y_data)
    y = np.where(y_data == 'No', 0, np.where(y_data == 'Yes', 1, -1))
    #['DIRECCION_VIENTO'] = df['DIRECCION_VIENTO'].astype(float)
    #df = df.replace(r'^\s*$', np.nan, regex=True)
    print(df.dtypes)
    df = df.astype(float)


    # x = df.iloc[0:20, 0:4]
    # x1 = df.iloc[0:20, 4:8]
    # x2 = df.iloc[0:20, 9:11]
    # x3 = df.iloc[0:20, 11:13]
    # x4 = df.iloc[0:20, 13:15]
    # print(x4)

    x = df.iloc[0:200, 1:13].values

    df['DIRECCION_VIENTO'] = pd.to_numeric(df['DIRECCION_VIENTO'], errors='coerce')

    training_data = list((x, y))
    print(training_data)


    # bpnn = network.Network(sizes = [13,4,1], eta = 0.01, epochs = 5, sizeBatch = 5)
    # bpnn.SGD(training_data)


def conditions_intensidad(x):
    if x == 'None':
        return 0.
    if x == 'Low':
        return 1.
    if x == 'Moderate':
        return 2.
    if x == 'High':
        return 3.
    else:
        return -1.

def conditions_estado_carretera(x):
    if x == 'Dry':
        return 0.
    if x == 'Snow covered':
        return 1.
    if x == 'Visible tracks':
        return 2.
    if x == 'Wet':
        return 3.
    else:
        return -1.

def conditions_tipo_precipitacion(x):
    if x == 'clear':
        return 0.
    if x == 'rain':
        return 1.
    if x == 'snow':
        return 2.
    else:
        return -1.