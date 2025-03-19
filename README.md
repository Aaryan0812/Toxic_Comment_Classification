# Multilingual Toxic Comment Classification

## 📌 Project Overview
The **Multilingual Toxic Comment Classification** project aims to detect and classify toxic comments on social media. Toxicity includes **hate speech, abusive language, and obscene content**.

This project leverages deep learning models such as **LSTM, LSTM-CNN, BERT, and GRU** to enhance classification performance across multiple languages.

## 📂 Dataset
The dataset used is the **Jigsaw Toxic Comment Classification** dataset from Kaggle, which includes multilingual data with annotated toxicity labels. The dataset contains comments and corresponding binary labels for different toxicity types.

🔗 Dataset Link: [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification)

## 🛠️ Technologies Used
- **Deep Learning Frameworks**: TensorFlow, PyTorch
- **Natural Language Processing (NLP)**: NLTK
- **Machine Learning Models**: LSTM, LSTM-CNN, BERT, GRU
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 🚀 Project Workflow
1. **Data Collection & Preprocessing**
   - Load the dataset and explore toxicity labels.
   - Clean text data (remove special characters, stopwords, etc.).
   - Tokenization and padding for deep learning models.
   
2. **Model Training & Evaluation**
   - Train multiple models (LSTM, LSTM-CNN, BERT, GRU) for classification.
   - Use word embeddings like Word2Vec, GloVe, or BERT embeddings.
   - Compare model performance using metrics like accuracy, F1-score, and AUC-ROC.
   
3. **Deployment & Testing**
   - Deploy the best-performing model.
   - Test predictions on real-world social media comments.

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-score**
- **AUC-ROC Curve**

## 📌 Future Improvements
- Fine-tune BERT for better multilingual toxicity detection.
- Implement real-time toxicity detection on social media comments.
- Integrate explainable AI techniques to understand model predictions.
- Expand dataset coverage for better generalization across languages.



