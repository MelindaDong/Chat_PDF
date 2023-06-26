# Description: This file contains functions that are used in the main.py file
# Author: Melinda Dong
from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet
import openai


# Function: extract_raw_text_from_pdf
def extract_raw_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Replace newline characters with spaces
    raw_text = raw_text.replace('\n', ' ')
    # Replace specific hyphenated cases with placeholders
    raw_text = re.sub(r'(\w)-\s(\w)', r'\1\2', raw_text)
    return raw_text

# Function: extract_in_text_citations and clean text
def extract_in_text_citations(text):
    # Define the citation patterns based on the two cases
    pattern_1 = r'\s?\[(?:\w+(?:\+)?\d{2}(?:,?\s)?)+\]'
    pattern_2 = r'\(\w+(?:\set\s?al.,)?\s?\d{4}[a-z]?(?:;\s?\w+(?:\set\s?al.,)?\s?\d{4}[a-z]?)*\)'
    pattern_3 =r'\([^()]*, \d{4}\)'

    # Find all matches of the citation patterns in the text
    citations_1 = re.findall(pattern_1, text)
    citations_2 = re.findall(pattern_2, text)
    citations_3 = re.findall(pattern_3, text)

    # Remove the citations from the original text to reduce noise
    cleaned_text = re.sub(pattern_1, '', text)
    cleaned_text = re.sub(pattern_2, '', cleaned_text)
    cleaned_text = re.sub(pattern_3, '', cleaned_text)

    return citations_1 + citations_2 + citations_3, cleaned_text

# get the context of reference
def get_context(text, target):
   
    pre_text = text.split(target)[0] 
    #print(f"pre_text: {pre_text} + '\n'")
    after_text = text.split(target)[1] 

    # get the previous 1 sentences
    pre_sentences = pre_text.split('.')[-1:] if len(pre_text.split('.')) >= 1 else []
    # get the next 1 sentences
    after_sentences = after_text.split('.')[:1] if len(after_text.split('.')) >= 1 else []
    context = ' '.join(pre_sentences + [target] + after_sentences)
    return context


#Process the review column line by line to do text preprocessing
def process_sentence(review):
    # remove the punctuations
    review = re.sub(r"[^\w\s]+", "", review)
    # convert the review to lower case
    review = review.lower()
    # remove the stopwords
    stop_words = set(stopwords.words('english'))
    # tokenize the words
    word_tokens = word_tokenize(review)
    filtered_review = [w for w in word_tokens if not w in stop_words]
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_review = [lemmatizer.lemmatize(w) for w in filtered_review]
    # return the processed review
    return lemmatized_review


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# calculate the unigram probability of a word in the corpus
def calculate_unigram_probability(corpus, word):
    word_count = corpus.count(word)
    total_words = len(corpus)
    unigram_probability = word_count / total_words
    return unigram_probability


def sentence_embedding(word_embeddings , sentences, a = 0.5, word_probabilities = None):
    sentence_embeddings = {}
    for index, s in sentences.items():
        vs = np.zeros(50)  # Initialize sentence embedding as zero vector
        for w in s:
            try:
                a_value = a / (a + word_probabilities[w])  # Smooth inverse frequency, SIF
                vs += a_value * word_embeddings[w] * (1/len(s)) # vs += sif * word_vector
                #vs += ((word_embeddings[w] * a)/(a + word_probabilities[w]))* (1/len(s))
            except KeyError:
                continue
        sentence_embeddings[index] = vs

    sentence_list = list(sentence_embeddings.values())
    num_sentences = len(sentence_list)
    embedding_dim = sentence_list[0].shape[0]  # Assuming all embeddings have the same dimension
    X = np.zeros((embedding_dim, num_sentences))

    for i, embedding in enumerate(sentence_list):
        X[:, i] = embedding

    # Perform singular value decomposition
    u, _, _ = np.linalg.svd(X, full_matrices=False)  #full_matrices=False ensures that only the necessary number of singular vectors is returned
    u = u[:, 0]  # Extract first singular vector

    for index, s in sentences.items():
        vs = sentence_embeddings[index]
        uuT = np.outer(u, u)  # Compute the outer product of u with itself
        vs = vs - np.dot(uuT, vs)  # Subtract the product of uuT and vs from vs
        sentence_embeddings[index] = vs

    return sentence_embeddings


def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


# get the top 3 similar chunks
def get_top3_similar_chunks(query, df_dict_vec):
    ranking = {}
    for q in df_dict_vec:
        ranking[q] = cosine_similarity(query, df_dict_vec[q])
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    if len(ranking) < 3:
        return ranking
    return ranking[:3]

# get the top 5 similar chunks
def get_top5_similar_chunks(query, df_dict_vec):
    ranking = {}
    for q in df_dict_vec:
        ranking[q] = cosine_similarity(query, df_dict_vec[q])
    ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    if len(ranking) < 5:
        return ranking
    return ranking[:5]





def ask_question(paragraph, question):
    chat_history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Here are some texts from a paper: ' + paragraph},
        {'role': 'assistant', 'content': question}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history
    )

    answer = response.choices[0].message.content
    return answer