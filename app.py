import re
import os
import sys
import flask
import base64
import torchvision.transforms as transforms

from io import BytesIO
from PIL import Image

# Append pix2pix filepath to app.py
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pix2pix')
if module_path not in sys.path:
    sys.path.append(module_path)

from convert import convert_image
from models import Pix2PixModel
from options import TestOptions
from flask import Flask, render_template, request, send_file, jsonify

# Global Vars
# CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opt = TestOptions().parse()
opt.checkpoints_dir = os.path.join('./pix2pix', opt.checkpoints_dir)

model = Pix2PixModel()
model.initialize(opt)

transformers = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

app = Flask(__name__)


# Routes
# Index
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api')
def api_page():
    return render_template('api.html')


# Helper function to serve PIL image
def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG')
    return base64.b64encode(img_io.getvalue())


# Generate pepe endpoint
@app.route('/generate', methods=['POST'])
def generate():
    global model, opt, transformers

    if 'img' in request.form:
        img_data = re.sub('^data:image/.+;base64,', '', request.form['img'])
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = img.resize((256, 256), Image.BILINEAR)
        img = convert_image(img, model, transformers)
        return jsonify({'img': serve_pil_image(img).decode('utf-8')})

    elif 'img' in request.json:
        img_data = re.sub('^data:image/.+;base64,', '', request.json['img'])
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = img.resize((256, 256), Image.BILINEAR)
        img = convert_image(img, model, transformers)
        return jsonify({'img': serve_pil_image(img).decode('utf-8')})

    return jsonify({'error': 'img not found'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, threaded=True)
