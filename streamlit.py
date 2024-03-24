import streamlit as st
from joblib import load

# Function to process the review based on the selected model
def process_review(review, model):
    # Dummy processing logic - replace with actual model prediction
    if model == "Model A":
        return "Model A prediction for '{}'".format(review)
    elif model == "Model B":
        return "Model B prediction for '{}'".format(review)
    elif model == "Model C":
        return "Model C prediction for '{}'".format(review)
    elif model == "KNN":
        return "KNN prediction for '{}'".format(review)
    elif model == "Logistic Regression":
        return "Logistic Regression prediction for '{}'".format(review)
    elif model == "Decision Tree":
        return "Decision Tree prediction for '{}'".format(review)
    elif model == "Random Forest":
        return "Random Forest prediction for '{}'".format(review)



def model_selection(text_processing_method,ml_model):
    if text_processing_method == "Regular":
        vectorizer = load('/tfidf1.joblib')
        if ml_model == "KNN":
            return (load('/model_r_knn.joblib'),vectorizer)
        elif ml_model == "Logistic Regression":
            return (load('/model_r_log.joblib'),vectorizer)
    elif text_processing_method == "Stemming":
        vectorizer = load('/tfidf2.joblib')
        if ml_model == "KNN":
            return (load('/model_s_knn.joblib'),vectorizer)
        elif ml_model == "Logistic Regression":
            return (load('/model_s_log.joblib'),vectorizer)
    else:
        vectorizer = load('/tfidf3.joblib')
        if ml_model == "KNN":
                return (load('/model_l_knn.joblib'),vectorizer)
        elif ml_model == "Logistic Regression":
                return (load('/model_l_log.joblib'),vectorizer)



def main():
    st.title("Text Processing Model Selector")

    # First select box for text processing method
    text_processing_method = st.selectbox(
        "Select text processing method:",
        ["Regular", "Stemming", "Lemmatization"]
    )
    # Second select box for model selection based on the text processing method
    method_selection = get_model_selection(text_processing_method)
    ml_model = st.selectbox("Select ML model:", method_selection)
    model,vector_model = model_selection(text_processing_method,ml_model)
    # Input box for user to enter review
    review = st.text_area("Enter your review here:")

    # Button to trigger model prediction
    if st.button("Predict"):
        if review:
            prediction = model.predict(vector_model.transform([review]))[0]
            st.write("Rating for this review is :", prediction)
        else:
            st.write("Please enter a review.")

def get_model_selection(text_processing_method):
    if text_processing_method == "Regular":
        return ["KNN", "Logistic Regression"]
    elif text_processing_method == "Stemming":
        return ["KNN", "Logistic Regression"]
    elif text_processing_method == "Lemmatization":
        return ["KNN", "Logistic Regression"]

if __name__ == "__main__":
    main()

