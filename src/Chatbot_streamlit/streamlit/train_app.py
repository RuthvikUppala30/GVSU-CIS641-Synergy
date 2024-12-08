import streamlit as st
import json
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents dataset
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lists for storing data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Preprocess the intents dataset
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower case words, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=250, batch_size=5, verbose=1)

# Save the model as a pickle file
with open('chatbot_model.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'words': words,
        'classes': classes,
        'intents': intents
    }, file)

print("Model trained and saved!")

# Streamlit App
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = np.zeros(len(words))
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def chatbot_response(text, model, words, classes, intents):
    bag = bag_of_words(text, words)
    res = model.predict(np.array([bag]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        for r in results:
            tag = classes[r[0]]
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

st.title("University Chatbot")
st.write("Ask me anything about the university! Type your question below.")

# Load model from pickle file
with open('chatbot_model.pkl', 'rb') as file:
    chatbot_data = pickle.load(file)

model = chatbot_data['model']
words = chatbot_data['words']
classes = chatbot_data['classes']
intents = chatbot_data['intents']

user_input = st.text_input("You:", key="user_input")

if user_input:
    response = chatbot_response(user_input, model, words, classes, intents)
    st.text_area("Bot:", value=response, height=100, max_chars=None, key=None)
