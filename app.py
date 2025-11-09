from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image, ImageOps
import io
import os  # ⬅️ أضفنا استيراد os

# تحميل النموذج والفئات
model = load_model("ocr_model.h5")
with open("classes.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# عكس القاموس: {0: "أ", 1: "ب", ...}
classes = {v: k for k, v in class_indices.items()}

app = Flask(__name__)
CORS(app)

# دالة للعثور على أي ملف صوتي في مجلد الحرف
def find_sound_file(letter):
    sound_dir = os.path.join('sounds', letter)
    if not os.path.exists(sound_dir):
        return None
    
    # البحث عن أي ملف في المجلد (أول ملف نجده)
    files = os.listdir(sound_dir)
    if files:
        return files[0]  # إرجاع أول ملف في المجلد
    return None

@app.route('/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('sounds', filename)

def preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = img.resize((128, 128))
    img = ImageOps.autocontrast(img)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "❌ لم يتم إرسال صورة"})

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))

    img = preprocess(img)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = np.max(preds)
    class_id = np.argmax(preds)
    
    print(f"🔍 الثقة: {confidence:.2f}, الحرف المتوقع: {classes.get(class_id, 'غير معروف')}")

    if confidence < 0.3:
        return jsonify({"success": False, "error": f"❌ لم يتم التعرف على الحرف بدقة كافية (الثقة: {confidence:.2f})"})
    else:
        letter = classes[class_id]
        sound_file = find_sound_file(letter)  # ⬅️ البحث عن أي ملف صوتي
        
        response_data = {
            "success": True, 
            "message": f"✅ الحرف هو: {letter} (الثقة: {confidence:.2f})",
            "letter": letter
        }
        
        # إذا وجدنا ملف صوتي، نضيفه للاستجابة
        if sound_file:
            response_data["sound_file"] = sound_file
        
        return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)