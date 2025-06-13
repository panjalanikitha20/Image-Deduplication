from flask import Flask, render_template, request, jsonify
import os
import signal
import sys
from utils import find_duplicates, delete_files, select_directory, compare_all_videos, extract_image_features, compare_images

app = Flask(__name__)

def signal_handler(signal, frame):
    print("\nComparison terminated.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_directory', methods=['GET'])
def browse_directory():
    directory = select_directory()
    return jsonify({'directory': directory})

@app.route('/deduplicate', methods=['POST'])
def deduplicate():
    target_file = request.files['target_file']
    file_extension = os.path.splitext(target_file.filename)[1].lower()

    allowed_image_extensions = {'.jpg', '.jpeg', '.png'}
    allowed_video_extensions = {'.mp4', '.avi', '.mov'}

    if file_extension in allowed_image_extensions:
        allowed_extensions = allowed_image_extensions
        target_filepath = os.path.join('uploads', target_file.filename)
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        target_file.save(target_filepath)

        directory = request.form['directory']
        files_to_check = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if os.path.splitext(file)[1].lower() in allowed_extensions
        ]

        results = []
        for file in files_to_check:
            feature_match = None
            comparison = compare_images(target_filepath, file)
            if comparison:
                hash_match, feature_match_value = comparison
                results.append({
                    "target": file,
                    "comparison_file": file,
                    "hash_match": hash_match,
                    "feature_match": feature_match_value,
                    "similarity": (hash_match + feature_match_value) / 2
                })
            
        if not results:
            return jsonify({"message": "No duplicates or similar images found."})

        return jsonify(results)

    elif file_extension in allowed_video_extensions:
        allowed_extensions = allowed_video_extensions
        target_filepath = os.path.join('uploads', target_file.filename)
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        target_file.save(target_filepath)

        directory = request.form['directory']
        results = compare_all_videos(directory)

        if not results:
            return jsonify({"message": "No duplicates or similar videos found."})

        return jsonify(results)
    
    else:
        return jsonify({"error": "Unsupported file type."}), 400

@app.route('/delete_duplicate', methods=['POST'])
def delete_duplicate():
    file_path = request.json['file']
    delete_files(file_path)
    return jsonify({"message": f"Deleted {file_path}"}), 200

if __name__ == '__main__':
    app.run(debug=True)