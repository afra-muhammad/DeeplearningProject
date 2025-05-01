import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

entropy_features = pd.read_csv('../output_data/entropy_features.csv')
tcp_features = pd.read_csv('../output_data/tcp_features.csv')
payloads = pd.read_csv('../output_data/payloads.csv')
labels = pd.read_csv('../output_data/labels.csv')

#Tokenization and Padding of Payloads (Text Data)
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(payloads)
vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Max sequence length (adjust as needed)
payload_sequences = tokenizer.texts_to_sequences(payloads)
payload_padded = pad_sequences(payload_sequences, maxlen=max_len, padding='post')

X_tcp, X_tcp_test, X_entropy, X_entropy_test, X_payload, X_payload_test, y_train, y_test = train_test_split(
    tcp_features, entropy_features, payload_padded, labels, test_size=0.2, random_state=42
)

# Input Layer for TCP Header Features
tcp_input = Input(shape=(144,))
tcp_dense = Dense(32, activation='relu')(tcp_input)
tcp_dense = Dense(16, activation='relu')(tcp_dense)

# Input Layer for Entropy Features
entropy_input = Input(shape=(10,))
entropy_dense = Dense(16, activation='relu')(entropy_input)
entropy_dense = Dense(8, activation='relu')(entropy_dense)

# 🔹 Input Layer for Payload (Text)
payload_input = Input(shape=(max_len,))
embedding_layer = Embedding(vocab_size, 128, input_length=max_len)(payload_input)
lstm_layer = LSTM(64, return_sequences=False)(embedding_layer)

# 🔹 Merge all features
merged = Concatenate()([tcp_dense, entropy_dense, lstm_layer])
merged_dense = Dense(32, activation='relu')(merged)
merged_dense = Dense(16, activation='relu')(merged_dense)
output = Dense(1, activation='sigmoid')(merged_dense)  # Sigmoid for binary classification

# 🔹 Define and Compile Model
model = Model(inputs=[tcp_input, entropy_input, payload_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 Model Summary
model.summary()

# Train Model
history = model.fit(
    [X_tcp, X_entropy, X_payload], y_train,
    validation_data=([X_tcp_test, X_entropy_test, X_payload_test], y_test),
    epochs=1, batch_size=32
)

model.save("model_10PF_K2.keras")