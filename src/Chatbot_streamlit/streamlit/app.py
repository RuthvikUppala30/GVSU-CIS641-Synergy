import streamlit as st
import json
import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the saved model, words, classes, and intents
with open('chatbot_model.pkl', 'rb') as file:
    chatbot_data = pickle.load(file)

model = chatbot_data['model']
words = chatbot_data['words']
classes = chatbot_data['classes']
intents = chatbot_data['intents']

# Function to preprocess user input
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = np.zeros(len(words))  # Ensure length matches training vocabulary
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict the response
def chatbot_response(text):
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

# Streamlit App
st.title("University Chatbot")
st.write("Ask me anything about the university! Type your question below.")

# Input text from user
user_input = st.text_input("You:", key="user_input")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Bot:", value=response, height=100, max_chars=None, key=None)
