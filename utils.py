import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import hamming
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import subprocess
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_image_features(img_path):
    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.resize((224, 224))
        img_data = np.array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = tf.keras.applications.vgg16.preprocess_input(img_data)
        vgg16_feature = model.predict(img_data)
        return vgg16_feature.flatten()
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def compare_images(image1, image2):
    image1_features = extract_image_features(image1)
    image2_features = extract_image_features(image2)

    if image1_features is None or image2_features is None:
        return 0, 0

    hamming_distance = 1 - hamming(image1_features, image2_features)
    return hamming_distance * 100, hamming_distance * 100

def compare_videos(video1_path, video2_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if cap1.get(cv2.CAP_PROP_FRAME_WIDTH) != cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or \
       cap1.get(cv2.CAP_PROP_FRAME_HEIGHT) != cap2.get(cv2.CAP_PROP_FRAME_HEIGHT):
        return 0, 0

    feature_match, frame_count = 0, 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        s = ssim(frame1_gray, frame2_gray)
        feature_match += s
        frame_count += 1

    cap1.release()
    cap2.release()

    if frame_count == 0:
        return 0, 0

    return (feature_match / frame_count) * 100, (feature_match / frame_count) * 100

def find_duplicates(target_file, files):
    duplicates = {} 
    ext = os.path.splitext(target_file)[1].lower()

    for file in files:
        if target_file == file:
            continue

        if ext in ['.jpg', '.jpeg', '.png']:
            hash_match, feature_match = compare_images(target_file, file)
        elif ext in ['.mp4', '.avi', '.mov']:
            hash_match, feature_match = compare_videos(target_file, file)
        else:
            continue

        if hash_match > 95 or feature_match > 95:
            duplicates.setdefault(target_file, []).append({
                'file': file,
                'hash_match': hash_match,
                'feature_match': feature_match
            })

    return duplicates

def delete_files(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

def select_directory():
    result = subprocess.run(['python', 'select_directory.py'], capture_output=True, text=True)
    if result.stdout:
        data = json.loads(result.stdout)
        return data['directory']
    return None

def get_all_video_paths(directory):
    video_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(root, file))
    return video_paths

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:  # Adjust frame sampling rate as needed
            frame = cv2.resize(frame, (224, 224))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features = gray.flatten() / 255.0
            frames_features.append(features)
        frame_count += 1
    cap.release()
    if not frames_features:
        raise ValueError(f"No frames could be extracted from {video_path}")
    return np.mean(frames_features, axis=0)

def compare_all_videos(directory):
    video_paths = get_all_video_paths(directory)
    features_dict = {}
    for video in video_paths:
        try:
            features = extract_video_features(video)
            features_dict[video] = features
        except Exception as e:
            print(f"Error processing {video}: {e}")

    comparisons = []
    for i, (video1, feat1) in enumerate(features_dict.items()):
        for video2, feat2 in list(features_dict.items())[i+1:]:
            similarity = cosine_similarity([feat1], [feat2])[0][0]
            comparisons.append((video1, video2, similarity))

    return sorted(comparisons, key=lambda x: x[2], reverse=True)