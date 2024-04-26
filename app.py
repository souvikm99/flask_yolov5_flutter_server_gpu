import subprocess
import json
from flask import Flask, request, jsonify, render_template, send_file, current_app, redirect, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import base64
from datetime import datetime
import torch

from PIL import Image
import cv2

#####HEMANTH IMPORTS#####
import sqlite3
# from models import Student


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['DETECTION_FOLDER'] = './detections'
app.config['DETECTION_FOLDER_OUTPUT_IMAGE'] = './detections/output/exp'

@app.route('/', methods=['GET'])
def index():
    return 'Server is running!'


def run_command(file_path, is_image=True):
    class_names = {
        '0': 'Apple',
        '1': 'Banana',
        '2': 'Orange'
    }

    output_path = os.path.join(app.config['DETECTION_FOLDER'], 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
    
    if is_image:
        command = [
            'python3', 'yolov5/detect.py',
            '--weights', 'AppleBananaOrange.pt',
            '--img', '640',
            '--conf', '0.25',
            '--source', file_path,
            '--project', output_path,
            '--name', 'exp',
            '--exist-ok',
            '--save-txt',
            '--save-conf',
        ]
    else:
        # Different command for videos
        # command = [
        #     'python3', 'yolov5/detect.py',
        #     '--weights', 'AppleBananaOrange.pt',
        #     '--img', '640',
        #     '--conf', '0.25',
        #     '--source', file_path,
        #     '--project', output_path,
        #     '--name', 'exp',
        #     '--exist-ok',
        #     # Video-specific flags if necessary, e.g., frame rate control
        # ]

        command = [
            'python3', 'track.py',
            '--yolo_weights', 'AppleBananaOrange.pt',
            '--img', '640',
            '--conf-thres', '0.25',
            '--source', file_path,
            '--output', f'{output_path}/exp',
            '--save-vid',
            '--save-txt',
            # Video-specific flags if necessary, e.g., frame rate control
        ]
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("YOLOv5 Output:", result.stdout)
        
        if is_image:
            base_name = os.path.basename(file_path)
            file_name, _ = os.path.splitext(base_name)
            txt_file_path = os.path.join(output_path, 'exp/labels', f'{file_name}.txt')
            counts = {}
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as file:
                    for line in file:
                        parts = line.split()
                        cls_id = parts[0]
                        cls_name = class_names.get(cls_id, 'Unknown Class')
                        if cls_name in counts:
                            counts[cls_name] += 1
                        else:
                            counts[cls_name] = 1
                counts_str = ', '.join([f'{key}: {value}' for key, value in counts.items()])
                return txt_file_path, counts_str
            else:
                return None, ''
        else:
            # Return the path to the processed video
            base_name = os.path.basename(file_path)
            file_name, _ = os.path.splitext(base_name)
            txt_file_path = os.path.join(output_path,'exp', f'{file_name}_counts.txt')
            counts = {}
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as file:
                    for line in file:
                        cls_name, count = line.strip().split(': ')
                        counts[cls_name] = int(count)
                counts_str = ', '.join([f'{key}: {value}' for key, value in counts.items()])
                media_path = os.path.join(output_path, base_name)
                return media_path, counts_str
            else:
                return None, ''

            # return os.path.join(output_path, 'exp', os.path.basename(file_path)), None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return None, ''



@app.route('/upload', methods=['POST'])
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
    result_path, counts_str = run_command(file_path, is_image=is_image)

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
    # The path where media files are stored might have been incorrect.
    file_path = os.path.join(current_app.config['DETECTION_FOLDER'], 'output/exp', filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Extract the file extension and determine the MIME type.
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if file_ext in ['jpg', 'jpeg', 'png']:
        mimetype = 'image/jpeg'
    elif file_ext in ['mp4', 'avi']:
        mimetype = 'video/mp4'
    else:
        return jsonify({'error': 'Unsupported file type'}), 415

    # Correctly send the file with the appropriate MIME type.
    return send_file(file_path, mimetype=mimetype)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)

