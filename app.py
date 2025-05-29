from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
model = load_model('model/model_covid.keras')  # Ruta a tu modelo

IMG_SIZE = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    uploaded_img_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Crear carpeta uploads si no existe
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            # Crear nombre único para la imagen subida
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(upload_folder, filename)

            # Guardar la imagen
            file.save(filepath)

            # Preprocesar imagen para el modelo
            img = image.load_img(filepath, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            # Predicción
            pred = model.predict(x)[0][0]

            # Lógica de decisión (ajusta según tu modelo)
            if pred > 0.5:
                result = 'COVID-19 Positivo'
            else:
                result = 'Normal'

            uploaded_img_path = filepath

    return render_template('index.html', result=result, image_path=uploaded_img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
