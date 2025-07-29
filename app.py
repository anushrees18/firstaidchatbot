import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

file_path = r'C:\Users\anush\OneDrive\New folder\Documents\firstaid\firstaid.xlsx'  
data = pd.read_excel(file_path)

X = data["Problem"].tolist()
y = data["Solution"].tolist()

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

problem_embeddings = semantic_model.encode(X, convert_to_tensor=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

def predict_solution_with_semantics(problem):
    problem_embedding = semantic_model.encode(problem, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(problem_embedding, problem_embeddings)

    most_similar_index = torch.argmax(similarity_scores).item()
    highest_similarity_score = similarity_scores[0][most_similar_index].item()

    if highest_similarity_score < 0.5: 
        return None, highest_similarity_score
    return y[most_similar_index], highest_similarity_score


st.title("First Aid Prediction Chatbot")
st.write("Enter a problem description to get the corresponding first aid solution.")


if 'past_searches' not in st.session_state:
    st.session_state.past_searches = []


user_input = st.text_input("Enter a problem description:")


col1, col2 = st.columns([1, 550]) 


with col2:
    if user_input:
        try:
            
            st.session_state.past_searches.append(user_input)

            if len(st.session_state.past_searches) > 10:
                st.session_state.past_searches = st.session_state.past_searches[-10:]

            predicted_solution, similarity_score = predict_solution_with_semantics(user_input)
            if predicted_solution is None:
                st.error("The input doesn't match any known problems. Please enter a valid problem.")
            else:
                st.write(f"**Problem:** {user_input}")
                st.write(f"**Predicted Solution:** {predicted_solution}")
                st.write(f"**Similarity Score:** {similarity_score:.2f}")
                st.success("Solution Found!")  # Display solution found after everything is written
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.sidebar.header("Recommendations / Suggestions")
if st.session_state.past_searches:
    for idx, past_search in enumerate(reversed(st.session_state.past_searches)):
        st.sidebar.write(f"{idx + 1}. {past_search}")
else:
    st.sidebar.write("No search history yet.")

st.markdown("---")

