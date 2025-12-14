# DISEASE_DETECTOR

## 🐮 About The Project

**DISEASE_DETECTOR** is an AI-powered web application designed to identify livestock diseases from images. Built with a robust **MobileNetV2** deep learning model and wrapped in a **Flask** backend, this tool provides rapid, on-site disease diagnosis to help protect herds and ensure animal health.

The frontend features a distinct **Modern Brutalism** design aesthetic—prioritizing raw functionality, high contrast, and a bold, industrial look.

### 🛠️ Tech Stack

*   **Core:** Python 3
*   **Deep Learning:** TensorFlow, Keras (MobileNetV2 architecture)
*   **Backend:** Flask
*   **Frontend:** HTML5, CSS3 (Brutalism Style), JavaScript
*   **Image Processing:** Pillow (PIL), NumPy

### ✨ Features

*   **Instant Analysis:** Upload an image and get immediate disease predictions.
*   **High Accuracy:** Utilizes a pre-trained MobileNetV2 model fine-tuned for livestock skin conditions.
*   **7 Detectable Classes:**
    *   Bovine Respiratory Disease
    *   Contagious Ecthyma
    *   Dermatitis
    *   Healthy
    *   Lumpy Skin Disease
    *   And more...
*   **Responsive Design:** Works on desktop and mobile devices.

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

*   Python 3.8 or higher
*   pip (Python package manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/im-Amrith/DISEASE_DETECTOR.git
    cd DISEASE_DETECTOR
    ```

2.  **Set up the Virtual Environment**
    It is recommended to use the provided virtual environment or create a new one.
    ```bash
    # Create a new venv (if not using venv1)
    python -m venv venv
    
    # Activate it
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirement.txt
    ```
    *Note: This project requires `tf_keras` for legacy model compatibility.*

---

## 🖥️ Usage

1.  **Navigate to the app directory**
    ```bash
    cd app
    ```

2.  **Run the Flask Server**
    ```bash
    python server.py
    ```

3.  **Access the Application**
    Open your web browser and go to:
    `http://127.0.0.1:5000`

4.  **Analyze an Image**
    *   Click **[ UPLOAD IMAGE ]**.
    *   Select a clear photo of the livestock skin condition.
    *   Click **ANALYZE NOW**.
    *   View the status report and prediction.

---

## 📂 Project Structure

```
DISEASE_DETECTOR/
├── app/
│   ├── trained_model/      # Contains the .h5 model file
│   ├── static/             # CSS and JavaScript files
│   ├── templates/          # HTML templates
│   ├── server.py           # Main Flask application
│   └── class_indices.json  # Mapping of class IDs to names
├── requirement.txt         # Python dependencies
└── README.md               # Project documentation
```

## ⚠️ Disclaimer

This tool is for educational and supportive purposes only. It should not replace professional veterinary diagnosis. Always consult with a qualified veterinarian for definitive treatment plans.
