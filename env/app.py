from flask import Flask, request, render_template
import pandas as pd
import joblib

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta para la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    form_data = request.form.to_dict()
    data = pd.DataFrame([form_data])

    # Convertir variables categóricas a dummies
    data = pd.get_dummies(data)

    # Cargar el modelo y el escalador
    modelo = joblib.load('../Modelo Regresion Logistica/modelo.pkl')
    scaler = joblib.load('../Modelo Regresion Logistica/scaler.pkl')
    x_columns = joblib.load('../Modelo Regresion Logistica/x_columns.pkl')

    # Alinear las columnas de entrada
    data = data.reindex(columns=x_columns, fill_value=0)

    # Escalar los datos
    data_scaled = scaler.transform(data)

    # Hacer la predicción
    prediction = modelo.predict(data_scaled)
    prediction_proba = modelo.predict_proba(data_scaled)

    probabilidad = prediction_proba[0][1]

    # Crear el mensaje basado en la predicción
    if prediction[0] == '<=50K':
        mensaje = 'Es probable que gane menos o igual a 50,000 dólares anuales en base a sus características.'
    else:
        mensaje = 'Es probable que gane más de 50,000 dólares anuales en base a sus características.'

    return render_template('result.html', mensaje=mensaje, probabilidad=probabilidad)


if __name__ == "__main__":
    app.run(debug=True)
