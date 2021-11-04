import numpy as np
import pandas as pd
import matplotlib.pyplot
import network
import netKeras

# Funciones que categorizan las columnas "Condiciones intensidad", "Estado carretera", "Tipo precipitación"
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

def accidente(x):
    if x== 'No':
        return 0.
    else:
        return 1.


#Convert excel to CSV to successful reading
train = pd.read_excel('Datos_PrActica_1_BPNN.xls')
train.to_csv('Datos_PrActica_1_BPNNenCSV.csv', index = False)
df = pd.read_csv('Datos_PrActica_1_BPNNenCSV.csv')

#Change the date and hour to just the hour and insert it on the first column of the frame
df.insert(1, "HORA", df['FECHA_HORA'].str[10:16], True)
df.drop('FECHA_HORA', inplace=True, axis=1)
# func = np.vectorize(getZona)
# zona_tiempo = func(df['HORA'])
# df.drop('HORA', inplace=True, axis=1)
# df.insert(1, "HORA",zona_tiempo, True)



#Vectorice any string or object value to an integer to avoid any issues
func = np.vectorize(conditions_tipo_precipitacion)
tipo_precipitacion = func(df['TIPO_PRECIPITACION'])
df.drop('TIPO_PRECIPITACION', inplace=True, axis=1)

df.insert(8, "TIPO_PRECIPITACION",tipo_precipitacion, True)


func1 = np.vectorize(conditions_intensidad)
intensidad_precipitacion = func1(df['INTENSIDAD_PRECIPITACION'])
df.drop('INTENSIDAD_PRECIPITACION', inplace=True, axis=1)
df.insert(9, "INTENSIDAD_PRECIPITACION", intensidad_precipitacion, True)


func2 = np.vectorize(conditions_estado_carretera)
estado_carretera = func2(df['ESTADO_CARRETERA'])
df.drop('ESTADO_CARRETERA', inplace=True, axis=1)
df.insert(12, "ESTADO_CARRETERA", estado_carretera, True)

#Output
func3 = np.vectorize(accidente)
accident = func3(df['ACCIDENTE'])
df.drop('ACCIDENTE', inplace=True, axis=1)
df.insert(13,"ACCIDENTE", accident, True)


df = df.replace(r'^\s*$', 0, regex=True)

df["TEMERATURA_AIRE"]=df["TEMERATURA_AIRE"].astype(float)
df["HUMEDAD_RRELATIVA"]=df["HUMEDAD_RRELATIVA"].astype(float)
df["DIRECCION_VIENTO"]=df["DIRECCION_VIENTO"].astype(float)
df["VELOCIDAD_VEHICULO"]=df["VELOCIDAD_VEHICULO"].astype(float)
df["LONGITUD_VEHICULO"]=df["TEMERATURA_AIRE"].astype(float)
df["VELOCIDAD_VIENTO"]=df["VELOCIDAD_VIENTO"].astype(float)
df["NUMERO_EJES"]=df["NUMERO_EJES"].astype(float)
df["CARRIL_CIRCULACION"]=df["CARRIL_CIRCULACION"].astype(float)
df["PESO_VEHICULO"]=df["PESO_VEHICULO"].astype(float)

# Normalización valores para que estén entre 0-1
df["NUMERO_EJES"]=df["NUMERO_EJES"]/10
df["PESO_VEHICULO"]=df["PESO_VEHICULO"]/63548
df["HUMEDAD_RRELATIVA"]=df["HUMEDAD_RRELATIVA"]/97
df["VELOCIDAD_VEHICULO"]=df["VELOCIDAD_VEHICULO"]/169
df["DIRECCION_VIENTO"]=df["DIRECCION_VIENTO"]/360
df["VELOCIDAD_VIENTO"]=df["VELOCIDAD_VIENTO"]/14.5
print(df.dtypes)





#print(df.dtypes)


# x = df.iloc[0:20, 0:4]
# x1 = df.iloc[0:20, 4:8]
# x2 = df.iloc[0:20, 9:11]
# x3 = df.iloc[0:20, 11:13]
# x4 = df.iloc[0:20, 13:15]
# print(x4)

# xc = df.iloc[0:2, 1:13]
xv= df.iloc[0:20000, 1:13].values

# print(xv)
y=df.iloc[0:20000,13]
# print(y)

#df['DIRECCION_VIENTO'] = pd.to_numeric(df['DIRECCION_VIENTO'], errors='coerce')

training_data = list((xv, y))
#print(training_data)

print("NUESTRO PROPIO BACKPROPAGATION")
bpnn = network.Network(sizes = [12,1,1], eta = 0.01, epochs = 10, sizeBatch = 20)
bpnn.SGD(training_data)



print("BACKPROPAGATION CON KERAS")


# Training and test sets
xTrain=df.iloc[0:45874,1:13]
yTrain=df.iloc[0:45874,13]
xTest=df.iloc[45875:df.shape[0],1:13]
yTest=df.iloc[45875:df.shape[0],13]


netK=netKeras.netKeras(50,0.01)
netK.train(xTrain,yTrain, xTest,yTest)


