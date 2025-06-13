#  Image Deduplication Web App

The **Image Deduplication Web App** is a Python-based tool that helps users detect and manage **duplicate or visually similar images** within a directory. It is built using **Flask** for the web interface, and leverages **deep learning (VGG16)**, **feature hashing**, and **pairwise comparison algorithms** to ensure robust and intelligent detection of redundancy across large image sets.

This tool is ideal for digital asset managers, researchers, photographers, and developers who want to maintain organized, clutter-free image repositories.

---

##  Features

-  **Duplicate/Similar Image Detection**  
  Upload an image and scan any local directory for visually similar files based on content, not just file metadata.

-  **VGG16-Based Deep Feature Extraction**  
  Uses pre-trained **VGG16** (from Keras) to extract high-level feature vectors that capture the essence of each image.

-  **Pairwise Similarity Comparison**  
  Performs **pairwise cosine similarity** comparisons between the uploaded image and every image in the selected folder, ensuring comprehensive and precise analysis.

-  **Hash Matching**  
  Adds an extra layer of verification using perceptual hashing and Hamming distance to catch near-duplicates even under minor edits or compression changes.

-  **Directory Selection Interface**  
  Users can browse and select any directory to scan for duplicates.

-  **One-Click Duplicate Deletion**  
  Results are displayed with options to remove identified duplicates directly.

-  **Web-Based Interface**  
  Built with Flask and HTML templates for simple interaction.

---

##  Technologies Used

- **Python 3**
- **Flask** – lightweight web framework
- **TensorFlow / Keras** – deep learning model (VGG16)
- **OpenCV** – image reading and processing
- **Pillow** – advanced image handling
- **SciPy / NumPy** – numerical operations
- **Scikit-Image** – structural similarity index (SSIM)
- **Cosine Similarity** – for vector-based pairwise comparison
- **Pairwise Algorithms** – compares the target image against each file in the directory

---

##  How It Works

1. **User uploads an image** via the web interface.
2. **Select a directory** to search for duplicates.
3. For each image in the directory:
   - Extract features using **VGG16**.
   - Calculate **cosine similarity** between the uploaded image and each target.
   - Optionally compute **hash-based similarity** (Hamming distance).
4. Store and return similarity scores.
5. Display matched images with % similarity and allow deletion if needed.

---

##  Project Structure

## ├── app.py # Flask app with endpoints
## ├── utils.py # Image comparison and feature extraction logic
## ├── templates/
 │ └── index.html # Web UI
## ├── uploads/ # Temporarily holds uploaded images
## ├── requirements.txt # Python dependencies


---

##  How to Run

 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-deduplication-app.git
cd image-deduplication-app

2. Install Dependencies
``` bash
pip install -r requirements.txt

3. Launch the App
```bash
python app.py
Visit http://127.0.0.1:5000 in your browser.

---

Pairwise Comparison Algorithm

The application performs **pairwise comparisons** between the uploaded image and every image in the selected directory to determine visual similarity.

  How It Works:
- Image features are extracted using the **VGG16** model’s `fc1` layer.
- For each pair (**uploaded_image**, **target_image**), the system computes the **cosine similarity** between their feature vectors:
  ```python
  similarity = cosine_similarity([input_features], [target_features])[0][0]
Store results where similarity exceeds a given threshold (e.g., 90%).
Optionally cross-check with Hamming distance using hash comparison for redundancy confirmation.
This approach ensures high accuracy, even when images are resized, slightly modified, or compressed differently.

---
Author
Panjala Nikitha
B.Tech IT – Anurag University
