# MLOps Face Recognition Pipeline

End-to-end MLOps pipeline for face recognition using scikit-learn DecisionTreeClassifier on the Olivetti faces dataset.

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow including:
- Machine Learning model training and testing
- Flask web application for predictions
- Docker containerization
- Kubernetes deployment with 3 replicas
- CI/CD automation with GitHub Actions

## 📊 Dataset & Model

- **Dataset**: Olivetti faces dataset (sklearn.datasets)
  - 400 images of 40 different people
  - 64x64 pixel grayscale images
  - Split: 70% training, 30% testing

- **Model**: DecisionTreeClassifier (scikit-learn)
  - Max depth: 10
  - Min samples split: 5
  - Saved as: `savedmodel.pth`

## 🌿 Branch Structure

### `main` Branch
- Initial repository setup
- README.md and .gitignore

### `dev` Branch
- `train.py` - Model training script
- `test.py` - Model evaluation script
- `.github/workflows/ci.yml` - CI/CD workflow
- Automated testing on every push

### `docker_cicd` Branch
- `app.py` - Flask web application
- `templates/index.html` - Web interface
- `Dockerfile` - Container configuration
- `kubernetes-deployment.yaml` - K8s deployment (3 replicas)

⚠️ **Note**: Branches are NOT merged as per assignment requirements.

## 🚀 Quick Start

### 1. Train the Model
```bash
git checkout dev
pip install -r requirements.txt
python train.py
python test.py
```

### 2. Run Flask Application
```bash
git checkout docker_cicd
python app.py
# Visit http://localhost:5000
```

### 3. Docker Deployment
```bash
docker build -t face-recognition-ml .
docker run -p 5000:5000 face-recognition-ml
```

### 4. Kubernetes Deployment
```bash
kubectl apply -f kubernetes-deployment.yaml
kubectl get pods  # Should show 3 replicas
kubectl get services
```

## 📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD
├── templates/
│   └── index.html           # Web UI
├── train.py                 # Model training
├── test.py                  # Model testing
├── app.py                   # Flask application
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── kubernetes-deployment.yaml  # K8s deployment
├── .dockerignore
└── .gitignore
```

## 🔄 CI/CD Pipeline

GitHub Actions automatically:
1. Checks out code
2. Sets up Python environment
3. Installs dependencies
4. Runs `train.py` to train model
5. Runs `test.py` to validate accuracy

Triggers: Push to `dev` branch

## 🐳 Docker Hub

Docker image: `[Docker Hub Username]/face-recognition-ml:latest`

## 🛠️ Technologies Used

- **ML Framework**: scikit-learn 1.7.2
- **Web Framework**: Flask 3.1.2
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Python**: 3.9+

## 📊 Model Performance

- Training Accuracy: ~53.93%
- Test Accuracy: ~35.83%

## 🌐 Web Interface Features

- Modern drag-and-drop image upload
- Real-time face recognition
- Prediction confidence scores
- Health check endpoint (`/health`)
- Responsive design

## 📝 API Endpoints

- `GET /` - Web interface
- `POST /predict` - Face recognition prediction
- `GET /health` - Health check (for K8s probes)

## 👨‍💻 Author

**Bhagya Laxmi Pandiyan**

---

**MLOps Major Assignment** - End-to-End Pipeline Implementation
