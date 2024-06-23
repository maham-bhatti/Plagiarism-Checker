import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def textprocess(text):
    """
    Preprocess the input text by tokenizing, lemmatizing, and removing stop words and punctuation.
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def cal_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts.
    """
    text1 = textprocess(text1)
    text2 = textprocess(text2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    vector1 = tfidf_matrix[0].toarray()[0]
    vector2 = tfidf_matrix[1].toarray()[0]
    
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    similarity = similarity * 100
    return similarity

text1 = "Hey, Did you notice any plagiarism."
text2 = "Are you there."

similarity_score = cal_similarity(text1, text2)

print(f"Similarity Score: {similarity_score:.2f} %")

