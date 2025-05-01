import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dff = pd.read_csv('../output_data/data_for_read.csv')
dff ['Payload_filled'] = dff['PayloadStrings'].fillna("<EMPTY>").astype(str)

dff_OP = dff[dff['Payload_filled'] != "<EMPTY>"].copy()

payloads_f1 = dff_OP['Payload_filled'].astype(str)  # getting the payload text
labels_f1 = dff_OP['Label']

#Tokenization and Padding of Payloads (Text Data)
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(payloads_f1)
vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Max sequence length (adjust as needed)
payload_sequences_f1 = tokenizer.texts_to_sequences(payloads_f1)
payload_padded_f1 = pad_sequences(payload_sequences_f1, maxlen=max_len, padding='post')

X_trainz, X_testz, y_trainz, y_testz = train_test_split(payload_padded_f1, labels_f1, test_size=0.2, random_state=42)

modelz = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

modelz.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modelz.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history_ltz = modelz.fit(
    X_trainz, y_trainz,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

modelz.save("model_NLP_K2_clean.keras")