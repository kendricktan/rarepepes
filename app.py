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
from flask import Flask, render_template, request, send_file

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
        # Prepare to convert base64 png to image file
        img_data = re.sub('^data:image/.+;base64,', '', request.form['img'])
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = convert_image(img, model, transformers)
        return serve_pil_image(img)

    return {'error': 'img not found'}


if __name__ == '__main__':
    app.run()
