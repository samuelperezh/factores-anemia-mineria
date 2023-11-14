# # 🚀 Despliegue con Streamlit - Factores de anemia
# 
# - Miguel Angel Vélez Suarez
# - Samuel Pérez Hurtado

#Importamos librerías básicas
import pandas as pd # manipulacion dataframes
import numpy as np  # matrices y vectores
import matplotlib.pyplot as plt #gráfica

#Cargamos el modelo
import pickle
filename = 'data/modelo-hiperparametrizado.pkl'
model,labelencoder,variables,min_max_scaler = pickle.load(open(filename, 'rb'))

# Se instala la librería de streamlit
# !pip install streamlit

#Se crea interfaz gráfica con streamlit para captura de los datos
import streamlit as st

st.title('Predicción de factores de anemia')

age_5yr = st.selectbox('Rango de edad', ['25-29', '30-34', '35-39', '20-24', '15-19', '40-44', '45-49'])
residence = st.selectbox('Tipo de residencia', ['Urban', 'Rural'])
wealth = st.selectbox('Índice de riqueza', ['Richest', 'Richer', 'Middle', 'Poorer', 'Poorest'])
hemoglobin = st.slider('Hemoglobina ajustada a la altura y si es fumador', min_value=18, max_value=160, value=50, step=1)
mosquito_net = st.selectbox('Uso de mosquitero', ['Yes', 'No'])
smokes = st.selectbox('Fumador', ['No', 'Yes'])
breastfeeding = st.slider('Minutos a los que se amamantó el hijo por primera vez', min_value=0, max_value=250, value=10, step=1)
fever = st.selectbox('¿Ha tenido fiebre en las dos últimas semanas?', ['No', 'Yes', "Don't know"])
supplements = st.selectbox('Está tomando suplementos de hierro?', ['No', 'Yes', "Don't know"])

datos = [[age_5yr, residence, wealth, hemoglobin, mosquito_net, smokes, breastfeeding, fever, supplements]]
data = pd.DataFrame(datos, columns=['Age in 5-year groups', 'Type of place of residence',
                                    'Wealth index combined',
                                    'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)',
                                    'Have mosquito bed net for sleeping (from household questionnaire)',
                                    'Smokes cigarettes', 'When child put to breast',
                                    'Had fever in last two weeks', 'Taking iron pills, sprinkles or syrup']) #Dataframe con los mismos nombres de variables

#Se realiza la preparación
data_preparada=data.copy()
data_preparada = pd.get_dummies(data_preparada, drop_first=False, dtype='int64')
data_preparada[['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)', 'When child put to breast']]= min_max_scaler.transform(data_preparada[['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)', 'When child put to breast']])
data_preparada.head()

#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()

#Hacemos la predicción
Y_fut = model.predict(data_preparada)
print(labelencoder.inverse_transform(Y_fut))