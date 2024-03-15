import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

df = 'detecciones_alevines.csv'
data = pd.read_csv(df, sep=',')

tensor_columns = ['confidence', 'X1', 'Y1', 'X2', 'Y2']
for column in tensor_columns:
    data[column] = data[column].str.extract(r'tensor\((.*?)\)').astype(float)
data['timestamp'] = pd.to_datetime(data['timestamp'])

#Calculo de centroides
data['X_center'] = (data['X1'] + data['X2']) / 2
data['Y_center'] = (data['Y1'] + data['Y2']) / 2

#Funcion para Calcular la distancia promedio entre alevines
def calculo_distancia_promedio(group):
    if len(group) > 1:
        coordinates = group[['X_center', 'Y_center']].values
        distances = pdist(coordinates, metric='euclidean')
        average_distance = distances.mean()
    else:
        average_distance = 0
    return average_distance

#Aplicacion de la funcion para cada timestamp
calculo_distancia_promedio = data.groupby('timestamp').apply(calculo_distancia_promedio)

#Clasificacion de distancias
lower_threshold = calculo_distancia_promedio.quantile(0.33)
upper_threshold = calculo_distancia_promedio.quantile(0.66)

def clasificacion_distancia(distance):
    if distance <= lower_threshold:
        return 'Agrupados'
    elif distance <= upper_threshold:
        return 'Distanciados'
    else:
        return 'Comiendocon'
    
Categorias_de_distancia = calculo_distancia_promedio.apply(clasificacion_distancia)
Categorias_de_distancia.head(100)

df_final = pd.DataFrame({
    'timestamp': calculo_distancia_promedio.index,
    'distancia_promedio': calculo_distancia_promedio.values,
    'clasificacion': Categorias_de_distancia.values
})
print(df_final.head())

#preparar y dividir los datos en conjunto de entrenamiento y prueba
x = df_final[['distancia_promedio']]
y = df_final['clasificacion']   

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

#Predecir las etiquetas para el conjunto de prueba
y_pred = knn.predict(x_test)

#Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report

#Matriz de confusion
cm = confusion_matrix(y_test, y_pred)
classes = ['Agrupados', 'Distanciados', 'Erraticos']

#graficar la matriz de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Matriz de Confusión')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.show()

#Graficar el histograma de las clases predichas
plt.figure(figsize=(10, 7))
sns.countplot(x= y_pred, palette='Blues')
plt.title('Histograma Clases Predichas')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(range(len(classes)), classes)
plt.show()