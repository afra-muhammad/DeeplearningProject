import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

payloads = pd.read_csv('../output_data/payloads.csv')
labels = pd.read_csv('../output_data/labels.csv')

#Tokenization and Padding of Payloads (Text Data)
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(payloads)
vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Max sequence length (adjust as needed)
payload_sequences = tokenizer.texts_to_sequences(payloads)
payload_padded = pad_sequences(payload_sequences, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(payload_padded, labels, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history_lt = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

model.save("model_NLP_K.keras")