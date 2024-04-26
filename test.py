from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import json
import subprocess
import datetime
import shutil
import random
import string
import chardet 
import base64
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
CORS(app)

#handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
#handler.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
#app.logger.addHandler(handler)

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
DETECT_OP_FOLDER = "DetectOP"
IMAGE_FILE_PATH = ''  # This will be dynamically updated

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_content_type(file_path):
    """Determine the content type based on the file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    return "video/mp4" if file_extension in ['.mp4', '.avi', '.gif'] else "image/jpeg"

def run_command(file_path):
    """Run a command to process the file and update the global IMAGE_FILE_PATH."""
    global IMAGE_FILE_PATH
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    temp_output_name = 'temp_output'

    file_extension = os.path.splitext(file_path)[1].lower()

    # Decide which command to run based on file type
    if file_extension in ['.jpg', '.jpeg', '.png']:  # Image files
        command = [
            "python", "yolov5/detect.py",
            "--weights", "best_good_bad_fruits.pt",
            "--source", file_path,
            "--project", DETECT_OP_FOLDER,
            "--name", temp_output_name,
        ]
    else:  # Video files
        command = [
            "python3", "track.py",
            "--yolo_weights", "yolov5s.pt",
            "--source", file_path,
            "--output", f"{DETECT_OP_FOLDER}/{temp_output_name}",
            "--save-vid",
            "--save-txt",
        ]


    try:
        subprocess.run(command, check=True)
        temp_output_path = os.path.join(DETECT_OP_FOLDER, temp_output_name)
        output_files = os.listdir(temp_output_path)
        final_output_path = move_and_rename_output_files(temp_output_path, output_files, timestamp)
        IMAGE_FILE_PATH = final_output_path
        print("output video - ", IMAGE_FILE_PATH)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {str(e)}")

def move_and_rename_output_files(temp_output_path, output_files, timestamp):
    """Move and rename output files from the temporary directory to the final destination."""
    output_video = '' # Declare it as global if it's going to be accessed outside this function

    for original_file_name in output_files:
        original_file_path = os.path.join(temp_output_path, original_file_name)
        file_extension = os.path.splitext(original_file_name)[1]

        # Check if '_count' is in the original file name
        if '_count' in original_file_name:
            final_output_name = f"{timestamp}_counts{file_extension}"
        else:
            final_output_name = f"{timestamp}{file_extension}"

        final_output_path = os.path.join(DETECT_OP_FOLDER, final_output_name)
        shutil.move(original_file_path, final_output_path)

        # Check if the file is a video and update output_video accordingly
        if file_extension in ['.mp4', '.avi', '.jpg']:
            output_video = final_output_path

    shutil.rmtree(temp_output_path)
    return output_video

def get_video_info(video_path):
    """Get video resolution and aspect ratio using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    # Find the video stream
    for stream in video_info['streams']:
        if stream['codec_type'] == 'video':
            return stream
    return None

def convert_video(input_path, output_path, max_width=1280, max_height=720):
    """
    Convert video to a lower resolution while preserving aspect ratio.
    """
    video_info = get_video_info(input_path)
    if not video_info:
        print("Could not get video info.")
        return False
    
    original_width = int(video_info['width'])
    original_height = int(video_info['height'])
    
    
    # Calculate the target dimensions while preserving aspect ratio
    aspect_ratio = original_width / original_height
    if original_width > original_height:  # Landscape or square
        target_width = min(original_width, max_width)
        target_height = int(target_width / aspect_ratio)
        
    else:  # Portrait
        target_height = min(original_height, max_height)
        target_width = int(target_height * aspect_ratio)

    # After calculating target_width and target_height
    target_width, target_height = adjust_resolution(target_width, target_height)

    
    # Construct the ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale={target_width}:{target_height}',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Video conversion successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")
        return False
    
def adjust_resolution(width, height):
    """Adjust the resolution to ensure width and height are divisible by 2."""
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    return width, height

def generate_unique_filename(original_filename):
    """Generate a unique filename based on the original filename and a random string."""
    base, ext = os.path.splitext(original_filename)
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    new_filename = f"{base}_{random_string}{ext}"
    return new_filename

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '' or not allowed_file(file.filename):
#         return jsonify({'error': 'No selected file or file type not allowed'})

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)

#     video_extensions = {'.mp4', '.avi'}
#     # Extract the file extension and check if it's a video
#     file_extension = os.path.splitext(filename)[1].lower()
#     if file_extension in video_extensions:
#         # Define output path for converted video
#         # When setting the output_path in convert_video or before calling it
#         output_filename = generate_unique_filename(f"converted_{filename}")
#         output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
#         # Convert the video
#         conversion_successful = convert_video(file_path, output_path)

#         if conversion_successful:
#             print("Video converted successfully.")
#             # Optionally, delete the original video file to save space
#             # os.remove(file_path)
#             # # Update file_path to point to the converted video
#             file_path = output_path
#             run_command(file_path)
#             return jsonify({'message': 'File uploaded successfully', 'filename': filename})
#         else:
#             return jsonify({'error': 'Failed to convert video'})
#     else:
#         run_command(file_path)
#         return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("Upload endpoint hit")

    if 'file' not in request.files:
        logging.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        logging.warning("No selected file or file type not allowed")
        return jsonify({'error': 'No selected file or file type not allowed'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        logging.info(f"File {filename} saved successfully")
    except Exception as e:
        logging.error(f"Failed to save file {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to save the file'}), 500

    video_extensions = {'.mp4', '.avi'}
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension in video_extensions:
        output_filename = generate_unique_filename(f"converted_{filename}")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        conversion_successful = convert_video(file_path, output_path)
        if conversion_successful:
            logging.info("Video converted successfully")
            os.remove(file_path)  # Clean up original video after conversion if needed
            file_path = output_path  # Update path to point to converted video
        else:
            logging.error("Failed to convert video")
            return jsonify({'error': 'Failed to convert video'}), 500

    try:
        run_command(file_path)
        return jsonify({'message': 'File uploaded and processed successfully', 'filename': filename})
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to process the file'}), 500

@app.route('/get-media', methods=['GET'])
def get_media():
    if not IMAGE_FILE_PATH:
        return jsonify({'error': 'No media available'})
    content_type = get_content_type(IMAGE_FILE_PATH)
    response = send_file(IMAGE_FILE_PATH, conditional=True)
    response.headers['Content-Type'] = content_type  # Correct way to set header
    return response


@app.route('/test-get-media', methods=['GET'])
def test_get_media():
    IMAGE_FILE_PATH_test = "/Users/souvikmallick/Desktop/MTP/flask_flutter_server/DetectOP/20240301092515.mp4"
    if not IMAGE_FILE_PATH_test:
        return jsonify({'error': 'No media available'})
    # content_type = get_content_type(IMAGE_FILE_PATH)
    response = send_file(IMAGE_FILE_PATH_test, conditional=True)
    # response.headers['Content-Type'] = content_type  # Correct way to set header
    return response

@app.route('/get-counts', methods=['GET'])
def get_counts():
    # Ensure IMAGE_FILE_PATH is defined and correctly points to your video file.
    counts_file_path = IMAGE_FILE_PATH.replace('.mp4', '_counts.txt').replace('.avi', '_counts.txt')
    try:
        with open(counts_file_path, 'rb') as file:
            data = file.read()
            print(f'counts {data}')
            try:
                # Try decoding as UTF-8
                text_data = data.decode('utf-8')
                print(f'counts {text_data}')
                return jsonify({'data': text_data})
            except UnicodeDecodeError:
                # Handle file as binary if UTF-8 decoding fails
                # Example: returning a message or handling binary data differently
                return jsonify({'error': 'File contains binary data, not text.'})
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)
