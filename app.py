import subprocess
import json
from flask import Flask, request, jsonify, render_template, send_file, current_app, redirect, Response, abort, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import base64
from datetime import datetime
import torch

from PIL import Image
import cv2

#####HEMANTH IMPORTS#####
import sys
sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from track2 import detect


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['DETECTION_FOLDER'] = './detections'
app.config['DETECTION_FOLDER_OUTPUT_IMAGE'] = './detections/output/exp'

model = torch.hub.load('yolov5','custom', path='AppleBananaOrange.pt',force_reload=True,source='local', device='0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model2 = DetectMultiBackend('AppleBananaOrange.pt', device=device, dnn=False)

def run(model, im):
    # Check the file extension to determine if it's an image or a video
    _, file_extension = os.path.splitext(im)
    if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        # Process as image
        results = model(im)
        results.save()
        detections = results.pandas().xyxy[0]
        # Count the occurrences of each class
        class_counts = detections['name'].value_counts().to_dict()
        # Convert the class counts dictionary to a formatted string
        class_counts_str = ', '.join(f"{cls}: {count}" for cls, count in class_counts.items())
        print("Detected objects count per class:", class_counts)
        print(results)
        return 'lo1', class_counts_str
    else:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
        # command = [
        #     'python3', 'track.py',
        #     '--yolo_weights', 'AppleBananaOrange.pt',
        #     '--img', '640',
        #     '--conf-thres', '0.25',
        #     '--source', im,
        #     '--output', app.config['DETECTION_FOLDER_OUTPUT_IMAGE'],
        #     '--save-vid',
        #     '--save-txt',
        #     # Video-specific flags if necessary, e.g., frame rate control
        # ]
        # result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print("YOLOv5 Output:", result.stdout)

        # # Return the path to the processed video
        # base_name = os.path.basename(im)
        # file_name, _ = os.path.splitext(base_name)
        # txt_file_path = os.path.join( app.config['DETECTION_FOLDER_OUTPUT_IMAGE'], f'{file_name}_counts.txt')
        # counts = {}
        # if os.path.exists(txt_file_path):
        #     with open(txt_file_path, 'r') as file:
        #         for line in file:
        #             cls_name, count = line.strip().split(': ')
        #             counts[cls_name] = int(count)
        #     counts_str = ', '.join([f'{key}: {value}' for key, value in counts.items()])
        #     media_path = os.path.join( app.config['DETECTION_FOLDER_OUTPUT_IMAGE'], base_name)
        #     return media_path, counts_str

        parser = argparse.ArgumentParser()
        parser.add_argument('--yolo_weights', nargs='+', type=str, default='AppleBananaOrange.pt', help='model.pt path(s)')
        parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str, default=im, help='source')
        parser.add_argument('--output', type=str, default='detections/output/exp', help='output folder')  # output folder
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
        parser.add_argument('--save-vid',default=True, action='store_true', help='save video tracking results')
        parser.add_argument('--save-txt', default=True, action='store_true', help='save MOT compliant results to *.txt')
        # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--evaluate', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
        parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        with torch.no_grad():
            detect(opt, model)

        base_name = os.path.basename(im)
        file_name, _ = os.path.splitext(base_name)
        txt_file_path = os.path.join( app.config['DETECTION_FOLDER_OUTPUT_IMAGE'], f'{file_name}_counts.txt')
        counts = {}
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                for line in file:
                    cls_name, count = line.strip().split(': ')
                    counts[cls_name] = int(count)
            counts_str = ', '.join([f'{key}: {value}' for key, value in counts.items()])
            media_path = os.path.join( app.config['DETECTION_FOLDER_OUTPUT_IMAGE'], base_name)
            return media_path, counts_str


# lol, lol2 = run(model2, 'uploads/20240427192156_1000447302.mp4')

@app.route('/', methods=['GET'])
def index():
    gpu_name = check_gpu()
    print("Using GPU:", gpu_name)
    return f'Server is running with GPU: {gpu_name}!'


def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_name = result.stdout.strip()
        return gpu_name
    except FileNotFoundError:
        return "No GPU available (nvidia-smi not found)"

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate a unique filename using current timestamp
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{current_time}_{secure_filename(file.filename)}"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(file_path)

    is_image = filename.endswith(('.jpg', '.jpeg', '.png'))
    #result_path, counts_str = run_command(file_path, is_image=is_image)
    if is_image:
        result_path, counts_str = run(model, file_path)
    else:
        result_path, counts_str = run(model2, file_path)

    if result_path:
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'filename': filename,
            'result_filepath': result_path,
            'counts': counts_str,
        })
    else:
        return jsonify({'error': 'Failed to process the file'})

@app.route('/get-media/<filename>', methods=['GET'])
def get_media(filename):
    base_filename = os.path.splitext(filename)[0]  # Strip any existing extension

    # Determine the new extension based on the original file type
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if file_ext in ['mp4', 'avi']:
        new_filename = base_filename + '.' + file_ext  # Keep original video extension
    elif file_ext in ['jpg', 'jpeg', 'png']:
        new_filename = base_filename + '.jpg'  # Change image extension to .jpg
    else:
        return jsonify({'error': 'Unsupported file type'}), 415

    # Construct the path with the new filename
    file_path = os.path.join(current_app.config['DETECTION_FOLDER'], 'output/exp', new_filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Determine the MIME type based on the file extension
    if file_ext in ['jpg', 'jpeg', 'png']:
        mimetype = 'image/jpeg'
    elif file_ext in ['mp4', 'avi']:
        mimetype = 'video/mp4'
    else:
        return jsonify({'error': 'Unsupported file type'}), 415

    # Correctly send the file with the appropriate MIME type.
    return send_file(file_path, mimetype=mimetype)


@app.route('/files', methods=['GET'])
def list_files():
    """Endpoint to list all files in the Modelzoo directory."""
    files = [file for file in os.listdir('Modelzoo') if os.path.isfile(os.path.join('Modelzoo', file))]
    return jsonify(files)

@app.route('/files/<filename>', methods=['GET'])
def download_file(filename):
    """Endpoint to download a specific file from the Modelzoo directory."""
    if os.path.exists(os.path.join('Modelzoo', filename)) and os.path.isfile(os.path.join('Modelzoo', filename)):
        return send_from_directory('Modelzoo', filename, as_attachment=True)
    else:
        abort(404, description="File not found")






#####################################D E E P S O R T################################################
####################################################################################################



#####################################D E E P S O R T################################################
####################################################################################################


if __name__ == '__main__':
    gpu_name = check_gpu()
    print("Using GPU:", gpu_name)

    app.run(debug=True, host='0.0.0.0', port=5050)

