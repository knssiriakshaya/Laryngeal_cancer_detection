#  Laryngeal Cancer Detection using SqueezeNet (Deep Learning + Flask)

This project is a deep learning-powered web application for detecting **laryngeal cancer** and related conditions from medical images. It uses a lightweight and fast **SqueezeNet** model integrated with a responsive **Flask** web interface. Users can upload an image and receive a classification result clearly indicating whether **cancer is detected** or **not detected**.

---

## 🌐 Web Interface 

> A clean, user-friendly interface for uploading and classifying laryngeal images.


---

## 🚀 Features

- Upload an image for real-time classification
- Powered by PyTorch and SqueezeNet 1.1
- Responsive web design with live prediction
- "No Cancer Detected" clearly shown for healthy predictions
- Lightweight model ideal for deployment

---

## 🧾 Model Classes

The model classifies uploaded images into one of the following:

| Label | Description                                      |
|-------|--------------------------------------------------|
| `IPCL` | Inverted Papilloma and Laryngeal Cancer         |
| `Le`   | Laryngeal Cancer (Primary)                      |
| `Hbv`  | Hepatitis B Virus (Related Condition)           |
| `He`   | Healthy / No Cancer Detected                    |

> If the prediction is `He`, the app shows **"No Cancer Detected"** and hides the class name. Otherwise, it shows **"Laryngeal Cancer Detected"** and the full class description.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Responsive Design)
- **Backend**: Python, Flask
- **Model**: SqueezeNet 1.1 (fine-tuned for 4 classes)
- **Deep Learning**: PyTorch, Torchvision

---

## 📁 Project Structure

```
laryngeal-cancer-detector/
├── app.py                # Flask app
├── model_loader.py       # Model loading and prediction
├── best_model.pth             # Trained SqueezeNet weights
├── templates/
│   ├── index.html        # Upload page
│   └── result.html       # Prediction result page
├── uploads/              # Folder for temporary image storage
└── README.md             # Project documentation
```

---

## 🧪 Setup Instructions


### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```text
flask
torch
torchvision
pillow
```

### 2. Add Your Trained Model

Place the trained SqueezeNet weights file as `best_model.pth` in the project root.

### 3. Run the Flask App

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ Example Output

| Input Image            | Model Prediction                         |
|------------------------|------------------------------------------|
| Healthy Larynx         | No Cancer Detected                       |
| Inverted Papilloma     | Laryngeal Cancer Detected (IPCL)         |
| Laryngeal Tumor        | Laryngeal Cancer Detected (Le)           |

---

## 📌 Future Enhancements

- Add confidence scores or heatmaps
- REST API version of the backend
- Cloud deployment (Render / Heroku / AWS)
- Extend model to classify more cancer types


