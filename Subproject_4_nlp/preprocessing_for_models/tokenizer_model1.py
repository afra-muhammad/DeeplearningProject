import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

payloads = pd.read_csv('../output_data/payloads.csv')

#Tokenization and Padding of Payloads (Text Data)
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(payloads)
vocab_size = len(tokenizer.word_index) + 1
max_len = 100  # Max sequence length (adjust as needed)
payload_sequences = tokenizer.texts_to_sequences(payloads)
payload_padded = pad_sequences(payload_sequences, maxlen=max_len, padding='post')

#numpy save for faster load into models
np.save("payload_padded.npy", payload_padded)
# Convert to DataFrame and save to make it human readable
pd.DataFrame(payload_padded).to_csv("../output_data/payload_padded.csv", index=False)