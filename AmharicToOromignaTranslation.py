import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Read data from files
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

amh_sentences = read_data(r'D:\Projects\Machine Learning Projects\amh.txt')
orm_sentences = read_data(r'D:\Projects\Machine Learning Projects\orm.txt')


# Tokenize input and output sentences
def tokenize_sentences(sentences):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences), tokenizer

amh_sequences, amh_tokenizer = tokenize_sentences(amh_sentences)
orm_sequences, orm_tokenizer = tokenize_sentences(orm_sentences)

orm_sequences, orm_tokenizer = tokenize_sentences(orm_sentences)

# Add special tokens '<start>' and '<end>' to the vocabulary
orm_tokenizer.word_index['<start>'] = len(orm_tokenizer.word_index) + 1
orm_tokenizer.word_index['<end>'] = len(orm_tokenizer.word_index) + 1

# Update the vocabulary size
orm_vocab_size = len(orm_tokenizer.word_index) + 1


# Pad sequences
max_seq_length = max(max(len(seq) for seq in amh_sequences), max(len(seq) for seq in orm_sequences))
amh_sequences_padded = pad_sequences(amh_sequences, maxlen=max_seq_length, padding='post')
orm_sequences_padded = pad_sequences(orm_sequences, maxlen=max_seq_length, padding='post')

# Define vocabulary sizes
amh_vocab_size = len(amh_tokenizer.word_index) + 1
orm_vocab_size = len(orm_tokenizer.word_index) + 1

# Define the model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(amh_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(orm_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(orm_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([amh_sequences_padded, orm_sequences_padded[:, :-1]], orm_sequences_padded[:, 1:], epochs=50, batch_size=64, validation_split=0.2)

# Define inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Translate function
def translate(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = orm_tokenizer.word_index['<start>']
    translated_sentence = ''
    
    # Define maximum length to prevent infinite loop
    max_length = 20
    
    for _ in range(max_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Get the index of the token with maximum probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Convert the token index to the corresponding word
        sampled_word = orm_tokenizer.index_word.get(sampled_token_index, '<unk>')
        
        # Break the loop if the end token is encountered
        if sampled_word == '<end>':
            break
        
        # Append the word to the translated sentence
        translated_sentence += sampled_word + ' '
        
        # Update the target sequence for the next iteration
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update the states for the next iteration
        states_value = [h, c]

    return translated_sentence.strip()


# Create the main application window
root = tk.Tk()
root.title("Machine Translation")

# Function to handle translation
def translate_text():
    input_text = input_text_widget.get("1.0", "end").strip()
    input_sequence = amh_tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
    translated_text = translate(input_sequence_padded)
    output_text_widget.delete("1.0", "end")
    output_text_widget.insert("1.0", translated_text)

# Create input text box
input_text_widget = scrolledtext.ScrolledText(root, width=50, height=5, wrap=tk.WORD)
input_text_widget.grid(row=0, column=0, padx=10, pady=10)

# Create translate button
translate_button = tk.Button(root, text="Translate", command=translate_text)
translate_button.grid(row=0, column=1, padx=10, pady=10)

# Create output text box
output_text_widget = scrolledtext.ScrolledText(root, width=50, height=5, wrap=tk.WORD)
output_text_widget.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
