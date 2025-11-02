import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL and download nltk data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Handle case if file has "intents" key
if "intents" in data:
    intents = data["intents"]
else:
    intents = data

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
tags = []
patterns = []

for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Train model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to answer that. Could you rephrase?"

counter = 0

# Streamlit main app
def main():
    global counter
    st.title("🎓 Career Guidance Chatbot")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Career Guidance Chatbot. Type a message below to begin your career exploration!")

        # Create log file if not exists
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")

        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"👤 User: {row[0]}")
                    st.text(f"🤖 Chatbot: {row[1]}")
                    st.text(f"🕒 {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found yet.")

    elif choice == "About":
        st.header("📘 About the Project")
        st.write("""
        The Career Guidance Chatbot uses Natural Language Processing (NLP) and 
        Machine Learning to help users explore career paths, learn resume tips, and 
        gain insights into various professions. 

        Technologies Used:
        - Python
        - Scikit-learn (Logistic Regression)
        - NLTK for tokenization
        - Streamlit for web interface
        - CSV for chat history storage
                 
        Key Features:
        - 500+ intents covering careers, resume guidance, and skill tips  
        - Interactive, user-friendly design  
        - Real-time AI-powered responses

                 By Team : Future Preview 
                 Team Leader: Jyothika
                 Team Member: Gayathri
                              Sneha
                              Bharathi
                              Ratna Kumari 
                 Class of CSE A third year  
        """)

# Run Streamlit app
if __name__ == "__main__":
    main()
