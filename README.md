# 🎵 Smart Music Genre Classification

A full-stack AI-powered web app that classifies the genre of any uploaded music file using deep learning and the GTZAN dataset.

![Demo Screenshot](frontend.png)

---

## 🚀 Features
- 🎧 Upload `.mp3` or `.wav` files
- 🤖 Deep Learning model trained on GTZAN Dataset
- 🎨 Stunning animated UI with audio playback
- 📜 Local prediction history via `localStorage`
- 🌐 Flask backend with librosa and Keras model

---

## 🔧 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/keshav6740/smart-music-genre-classification.git
cd smart-music-genre-classification
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🎓 Model Training

> Optional: Skip this if you're using the pre-trained `.keras` model

```bash
python train.py
```

The training uses `genres_original/` audio files and creates `music_genre_classifier.keras`.

---

## 🧠 Run the Flask Backend
```bash
python server.py
```
> Flask server will start at: http://localhost:5000

---

## 🌐 Serve the Frontend
Use Python HTTP server to serve `index.html`:
```bash
python -m http.server 8000
```
Then open: [http://localhost:8000](http://localhost:8000)

---

## 📂 Folder Structure
```
keshav6740-smart-music-genre-classification/
├── README.md
├── index.html
├── pred.py
├── server.py
├── train.py
├── requirements.txt
└── Data/
    └── readme.md
      
```

---

## 💾 Prediction History
- Saved in `localStorage`
- Accessed via the floating "History" button
- Shows filename, genre, and timestamp

---

## 🧠 Model
- Built using TensorFlow + Keras Sequential API
- Uses Mel Spectrograms via `librosa`
- Trained on GTZAN dataset (10 genres)

---

## 📊 Dataset
See [`Data/readme.md`](Data/readme.md) for full GTZAN dataset details.

---

## Deploy On Render

This repo now includes:
- `Dockerfile` (installs Python deps + `ffmpeg`, runs `gunicorn`)
- `render.yaml` (Render Blueprint for one web service)

### Option 1: Blueprint (recommended)
1. Push this repo to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select this repo and create the service.

### Option 2: Manual Web Service
1. In Render, create a **Web Service** from this repo.
2. Environment: **Docker**.
3. Render will auto-detect and use the `Dockerfile`.

### Required env vars
- `PORT` is provided automatically by Render.
- (Optional) `LASTFM_API_KEY` if you want recommendations enabled.

