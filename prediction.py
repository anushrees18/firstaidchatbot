import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, util
import torch


file_path = r'C:\\Users\\anush\\OneDrive\\New folder\\Documents\\firstaid\\firstaid\\firstaid\\firstaid.xlsx'  
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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


def predict_solution_with_semantics(problem):
    problem_embedding = semantic_model.encode(problem, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(problem_embedding, problem_embeddings)

    most_similar_index = torch.argmax(similarity_scores).item()
    highest_similarity_score = similarity_scores[0][most_similar_index].item()
    
    if highest_similarity_score < 0.5:  
        return None, highest_similarity_score
    return y[most_similar_index], highest_similarity_score


while True:
    user_input = input("Enter a problem (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    try:
        predicted_solution, similarity_score = predict_solution_with_semantics(user_input)
        if predicted_solution is None:
            print("The input doesn't match any known problems. Please enter a valid problem.")
        else:
            print("Problem:", user_input)
            print("Predicted Solution:", predicted_solution)
    except Exception as e:
        print("Error:", str(e))
