![image](https://github.com/user-attachments/assets/3faaa052-d824-4dce-bb87-70ba5d31b51b)

# 🍽️ Restaurant Review Sentiment Analysis with Ensemble Learning

A robust sentiment analysis system that classifies restaurant reviews as positive or negative using an ensemble of machine learning and deep learning models.

[Live Demo](https://sultanmr-sentimental-analysis-app-941vjx.streamlit.app/)
![image](https://github.com/user-attachments/assets/528ff651-cdda-4619-a86c-3ce064661f61)

## 🌟 Key Features

- **Multi-Model Ensemble** combining:
  - 📊 Traditional ML (TF-IDF + Logistic Regression/SVM)
  - 🧠 Keras-based CNN
  - 🔥 PyTorch-based CNN
  - 🗳️ **Voting System** (hard and soft voting options)
  - 📈 Comprehensive performance metrics
  - 🛠️ Modular architecture for easy extension

## 📊 Performance Comparison

| Model                     | Accuracy | F1-Score | Training Time |
|---------------------------|----------|----------|---------------|
| Logistic Regression       | 0.81     | 0.81     | 30s           |
| Keras CNN                 | 0.71     | 0.84     | 2min          |
| PyTorch CNN               | 0.86     | 0.85     | 3min          |
| Customize Bert            | 0.89     | 0.84     | 2min          |
| Hugginface Transformer    | 0.94     | 0.85     | 0min          |
| **Ensemble (Voting)**     | **0.88** | **0.87** | -             |

## 🚀 Quick Start

### EDA
[Google collab file for initial data exploration](https://colab.research.google.com/drive/1Tckzj_MdDL2tnHsEpLQMFAizPzvm0NW0?usp=sharing)

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
git clone https://github.com/sultanmr/sentimental-analysis.git
cd sentimental-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application

```bash
# Train all models and evaluate performance
python main.py

# Launch interactive demo (requires Streamlit)
streamlit run app.py

# View models performance and confusion matrix (requires Tensorboard)
tensorboard --logdir logs
```
## 🏗️ Project Structure

```
sentiment-analysis/
│
├── logs/                    # Tensorboard logs for accuracy, loss and confusion matrix
├── models/                  # Model implementations code
├── EDA.ipynb                # Google collab file for initial data exploration
├── save/                    # Models pickle files
├── app.py                   # Streamlit web application
├── main.py                  # Training models
├── testm.py                 # Model testing
├── tb_utils.py              # Tensorbaord wrapper file
├── requirements.txt         # Dependencies
├── config.yaml              # Online learned models path
└── README.md                # This file
```

## 📚 Dataset

The dataset contains 1000 restaurant reviews with binary sentiment labels:

| Column   | Description                          |
|----------|--------------------------------------|
| Review   | Text of customer review              |
| Liked    | Sentiment label (1=Positive, 0=Negative) |

**Download:** [Restaurant_Reviews.tsv](https://www.dropbox.com/scl/fi/6mvhmvbuyijpt5rwzk12o/Restaurant_Reviews.tsv?rlkey=31dhfnze1subkcsdoa50irtvc&st=77nhe6hr&dl=1)

## 🧠 Model Architecture

### 1. Traditional Machine Learning
```mermaid
graph TD
    A[Raw Text] --> B[Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Logistic Regression]
    D --> E[Prediction]
```

### 2. Keras CNN
```mermaid
graph TD
    A[Raw Text] --> B[Tokenization]
    B --> C[Embedding Layer]
    C --> D[Conv1D + MaxPooling]
    D --> E[Flatten]
    E --> F[Dense Layer]
    F --> G[Prediction]
```

### 3. PyTorch CNN
```mermaid
graph TD
    A[Input Text] --> B[Tokenization]
    B --> C[Embedding Layer]
    C --> D[Conv1D Layer<br>Filters: 100, Kernel: 3]
    D --> E[ReLU Activation]
    E --> F[AdaptiveMaxPool1D]
    F --> G[Dropout Layer<br>p=0.5]
    G --> H[Fully Connected Layer]
    H --> I[Output Prediction]
```
### 4. Custom BERT Architecture
```mermaid
graph TD
    A[Input Text] --> B[DistilBERT Tokenizer]
    B --> C[Input IDs]
    B --> D[Attention Mask]
    C --> E[DistilBERT Layer]
    D --> E
    E --> F[Reshape Layer]
    F --> G[Conv1D<br>Filters:64, Kernel:2]
    G --> H[ReLU + Dropout]
    H --> I[Conv1D<br>Filters:128, Kernel:2]
    I --> J[ReLU + Dropout]
    J --> K[GlobalMaxPooling1D]
    K --> L[Dense Layer<br>Units:64]
    L --> M[Sigmoid Output]
```

### 5. Ensemble Voting
```mermaid
graph TD
    A[Input Review] --> B[Model 1]
    A --> C[Model 2]
    A --> D[Model 3]
    B --> E[Voting System]
    C --> E
    D --> E
    E --> F[Final Prediction]
```

## 📊 Results Visualization

![image](https://github.com/user-attachments/assets/fe79723c-7dda-4341-be34-a63916c112a5)
![image](https://github.com/user-attachments/assets/7d6c9e63-abfc-4a0d-90eb-59e71229d18b)
![image](https://github.com/user-attachments/assets/bb2abc85-2c30-48ae-9a01-eafc0f3cb3c5)


## 🛠️ Customization

### Training Configuration
Edit `.env` to modify:
- Model training local path
- CSV Path
- Training epochs
Edit `config.yaml` to modify:
- Model online URL

### Adding New Models
1. Create new model file in `models/` directory
2. Implement required interface:
   ```python
   def train(X_train, y_train)
   def predict(X_test)
   ```
3. Register model in `main.py` and config.yaml

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## ✉️ Contact

For questions or suggestions, please contact:
- [Sultan](mailto:sultanmr@hotmail.com)
- [Project Website](https://www.sultanmahmood.com)

---


Key improvements:
1. Add voice to text features for testing purposes
2. Add multilingual options
3. Added performance comparison table
4. Included customization instructions

