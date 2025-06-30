from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import numpy as np
import io

# loading class names
CLASS_NAMES = ['Grad de risc scăzut', 'Grad de risc crescut', 'Piele curată']  

app = Flask(__name__)

# loading the model
MODEL_PATH = './model2.h5'
model = load_model(MODEL_PATH, compile=False)

# adjusting input size to match the model 
IMG_SIZE = (180, 180)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    image_array = np.expand_dims(image, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()
    try:
        # # previewing the original image
        # image = Image.open(io.BytesIO(image_bytes))
        # image.show(title="Original")

        img_array = preprocess_image(image_bytes)  # shape (1, 180, 180, 3)

        # # removing batch dimension and converting back to PIL for preview
        # processed_image = Image.fromarray(img_array[0].astype('uint8'), 'RGB')
        # processed_image.show(title="Processed (Resized)")

        preds = model.predict(img_array)
        class_id = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        result = {
            'predicted_class_id': int(class_id),
            'predicted_class_name': CLASS_NAMES[class_id],
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
