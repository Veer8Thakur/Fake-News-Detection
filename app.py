import streamlit as st
import pickle
import pandas as pd

# Load the saved model + vectorizer
with open("logistic_regression_tfidf.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    vectorizer = data["vectorizer"]

# App title
st.title("📰 Fake News Detection App")

# Sidebar with simple explanation
st.sidebar.title("About This App")
st.sidebar.write(
    "This tool helps you check whether a news headline or article is likely to be "
    "real or fake. Just paste the text into the box and click Predict. "
    "The app will analyze the words and give you a result along with a probability score. "
    "Purpose: to make it easier for anyone to spot misleading information online."
)

# User input
user_input = st.text_area("Enter a news headline or article:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Transform input using the vectorizer
        transformed_input = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0][1]

        # Layout with columns
        col1, col2 = st.columns(2)
        if prediction == 1:
            col1.error("⚠️ Prediction: Fake News")
        else:
            col1.success("✅ Prediction: Real News")

        col2.metric("Probability (Fake)", f"{probability:.2f}")

        # Probability bar chart
        probs = pd.DataFrame({
            "Class": ["Real", "Fake"],
            "Probability": [1 - probability, probability]
        })
        st.bar_chart(probs.set_index("Class"))