# 🍽️ Restaurant Review Sentiment Analysis with Ensemble Voting

This project implements a sentiment analysis system that classifies restaurant reviews as **positive** or **negative** using an ensemble of three different models:

* 📊 Traditional Machine Learning (TF-IDF + Logistic Regression / SVM)
* 🧠 Keras-based Convolutional Neural Network (CNN)
* 🔥 PyTorch-based Convolutional Neural Network (CNN)

By combining diverse modeling techniques, the ensemble aims to achieve superior performance using both **soft** and **hard voting** strategies.

---

## 🚀 Features

* Preprocessing using NLTK with lemmatization and stopword removal.
* TF-IDF vectorization for traditional ML models.
* Keras and PyTorch CNNs with embedded text representations.
* Soft and hard ensemble voting strategies.
* Evaluation with accuracy, F1-score, and confusion matrix.
* Highly modular code structure for easy extension.

---

## 🧰 Tech Stack

* Python 3.8+
* Pandas, NumPy, Seaborn, Matplotlib
* NLTK for NLP preprocessing
* Scikit-learn for ML models
* TensorFlow/Keras for deep learning
* PyTorch for alternative deep learning modeling

---

## 📂 Project Structure

```
restaurant-review-ensemble/
│
├── data/                     # Downloaded datasets
├── models/                   # Saved model checkpoints (optional)
├── notebooks/                # Jupyter notebooks (if applicable)
├── restaurant_sentiment.py   # Main model pipeline (class-based)
├── README.md                 # Project overview
└── requirements.txt          # Required packages
```

---

## 📊 Dataset

* Source: [Restaurant\_Reviews.tsv (Dropbox)](https://www.dropbox.com/scl/fi/6mvhmvbuyijpt5rwzk12o/Restaurant_Reviews.tsv?rlkey=31dhfnze1subkcsdoa50irtvc&st=77nhe6hr&dl=1)
* Format: TSV file with two columns:

  * `Review`: Text of the review
  * `Liked`: Binary label (1 = Positive, 0 = Negative)

---

## 🏗️ How It Works

1. **Preprocessing**:

   * Traditional models: Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize.
   * Deep learning models: Tokenize and pad sequences.

2. **Model Training**:

   * **Traditional**: TF-IDF + Logistic Regression (or LinearSVC).
   * **Keras CNN**: Embedding + Conv1D + MaxPooling + Dense.
   * **PyTorch CNN**: Custom CNN with embedding, conv layers, and dropout.

3. **Ensemble Prediction**:

   * **Soft Voting**: Average predicted probabilities.
   * **Hard Voting**: Majority vote from binary predictions.

---

## 🧪 Evaluation Metrics

* Accuracy
* F1 Score
* Confusion Matrix
* Class-wise performance comparison

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/restaurant-review-ensemble.git
   cd restaurant-review-ensemble
   ```

2. **Create a Virtual Environment & Install Dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Project**
   You can use the `RestaurantSentimentAnalyzer` class in your Python script or notebook:

   ```python
   from restaurant_sentiment import RestaurantSentimentAnalyzer

   analyzer = RestaurantSentimentAnalyzer()
   df = analyzer.load_data()
   ```

---

## 📈 Sample Output

```
Best Traditional Model: LogisticRegression with n-gram=2, F1=0.847
Keras CNN training completed
PyTorch CNN training completed

Ensemble Accuracy (Soft Voting): 89.4%
```

---

## 🧠 Future Enhancements

* Incorporate LSTM and BERT-based models
* Include explainability tools (e.g., SHAP)
* Web app deployment using Streamlit or Flask
* Hyperparameter tuning using Optuna or Ray Tune

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [NLTK](https://www.nltk.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)
* [Scikit-learn](https://scikit-learn.org/)
* Inspired by ensemble learning and real-world NLP challenges.