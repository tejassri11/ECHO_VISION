

---

# **Vision Assistant For Blind Person**

An advanced **AI-powered vision assistant** that can **describe scenes**, **read text**, and **detect currency** from a live camera feed.
The project integrates **Computer Vision**, **Deep Learning**, and **Real-Time Processing** using:

* **BLIP** → Image captioning
* **MiDaS** → Depth estimation
* **PaddleOCR** → Text detection & extraction
* **Custom Currency Detector** → Indian currency classification
* **Flask** → Backend API
* **Web UI (HTML/JS)** → Live camera streaming + interactive controls
* **OpenCV** → Frame handling & processing
* **pyttsx3** → Text-to-Speech output

---

## 🚀 **Features**

### 🔹 **1. Scene Description (BLIP Model)**

* Generates human-like captions for any object or environment in front of the camera.
* Powered by **BLIP (Bootstrapping Language-Image Pretraining)**.
* Gives highly accurate and contextual descriptions.

### 🔹 **2. Depth Estimation (MiDaS)**

* Predicts relative depth for everything in the scene.
* Produces a heatmap-style depth view.

### 🔹 **3. OCR – Text Reading (PaddleOCR)**

* Detects and reads text from real-world objects (books, labels, signs, screens).
* Supports:

  * Multi-line text
  * Confidence scoring
  * Angle correction

### 🔹 **4. Currency Detection**

* Identifies Indian currency notes.
* Gives denomination & confidence score.
* Useful for visually impaired or general automation.

### 🔹 **5. Real-Time Processing via Flask**

* Live video feed from the browser.
* Captured frames are sent to Flask API for inference.
* Response displayed instantly on UI.

### 🔹 **6. Text-to-Speech Output**

* Every result (caption / text / currency) is spoken aloud.
* Uses **pyttsx3** for offline TTS.

---

## 🧠 **Tech Stack**

### **Backend**

* Python
* Flask
* BLIP (Transformers)
* MiDaS (Torch Hub)
* PaddleOCR
* OpenCV
* TensorFlow/Keras (Currency Model)
* pyttsx3

### **Frontend**

* HTML
* CSS
* JavaScript
* Web Speech API
* getUserMedia (Webcam Access)

---

## 🗂️ **Project Structure**

```
vision-assistant/
│
├── server.py                # Main Flask server
├── scene_describer.py       # BLIP caption + MiDaS depth
├── text_reader.py           # PaddleOCR-based text detection
├── currency_detector.py     # Currency classification module
├── static/
│   ├── script.js            # Frontend JS
│   ├── style.css            # Styling
│   └── index.html           # Main UI
└── models/
    └── currency_model.h5    # Custom-trained currency model
```

---

## ⚙️ **Installation**

### **1️⃣ Create Virtual Environment**

```bash
python -m venv vision_env
```

### **2️⃣ Activate Environment**

Windows:

```bash
vision_env\Scripts\activate
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ **Run the Application**

Start Flask server:

```bash
python server.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧪 **How It Works**

### 1️⃣ User opens web app → camera feed starts

### 2️⃣ User selects:

* **Describe Scene**
* **Read Text**
* **Detect Currency**

### 3️⃣ Frame sent to Flask backend

### 4️⃣ Backend runs:

* BLIP → Caption
* MiDaS → Depth
* PaddleOCR → Text
* Custom Model → Currency

### 5️⃣ Result returned + spoken aloud

## Preview
<img width="1873" height="954" alt="Screenshot 2025-11-21 002833" src="https://github.com/user-attachments/assets/62439d9b-677d-47ce-9975-a7bc67f43e47" />
<img width="1616" height="961" alt="Screenshot 2025-11-21 002138" src="https://github.com/user-attachments/assets/d9a97980-d7eb-4f15-bf9b-20720b1f782e" />
<img width="1882" height="904" alt="Screenshot 2025-11-21 003527" src="https://github.com/user-attachments/assets/4c0cb05f-965f-4b26-b75c-158eeadfea7f" />



---
