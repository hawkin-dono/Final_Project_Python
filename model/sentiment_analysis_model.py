import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import Precision, Recall, F1Score
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Embedding, BatchNormalization, Dropout
from keras_preprocessing.text import tokenizer_from_json
import seaborn as sns

class Sentiment_analysis_model:
    def __init__(self, weight_path= 'Data\model_data\weight2.h5' , tok_path = 'Data\model_data\\tokenizer.json', max_len =10000):
        self.weight_path = weight_path
        self.tok_path = tok_path
        self.max_len= max_len
        self.model = model = Sequential([
                Embedding(10000, 32),
                LSTM(32, activation= 'sigmoid',),
                Dense(3, activation='softmax')
        ])
        
        with open(self.tok_path, 'r', encoding='utf-8') as json_file:
            loaded_tok_json = json_file.read()
        self.tok = tokenizer_from_json(loaded_tok_json)
        
        model.load_weights(self.weight_path)
    def predict(self, data):
        
        sequences = self.tok.texts_to_sequences(data)
        sequences_matrix = pad_sequences(sequences, maxlen= self.max_len)
        
        self.model.load_weights(self.weight_path)
        
        y_pred_prob = self.model.predict(sequences_matrix)
        y_pred = np.argmax(y_pred_prob, axis= 1)
        
        return y_pred
    

def main():
    text = ['Tất cả anh em trong hang chú ý!!!!', '1 năm rồi đấy, nhanh thật',
       'Trong hang chán quá nên nghịch 1 tí 🤣 Chờ ngày ra ngoài hóng nắng 🔴🟡',
       'Ảo Malaysia :))))',
       '1. David Moyse: tạt tạt tạt tạt => không bài vở gì\n2. Van Gaal: chuyền qua chuyền lại => không bài vở gì\n3. Mourinho: thực dụng, cá tính mạnh']
    
            
    weight_path = 'Data\model_data\weight2.h5' 
    tok_path = 'Data\model_data\\tokenizer.json'
    model = Sentiment_analysis_model(weight_path, tok_path)
    
    res = model.predict(text)
    print(res)
    
if __name__ == '__main__':
    main()
        
            
    