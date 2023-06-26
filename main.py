print("Computer: Welcome to the chatPDF, homemade, very first version!\n")

import os
import functions
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key  # Replace with your actual API key

# load the pre-trained glove word embeddings
embeddings_dict = {}
with open("glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector


def check_file_exists(filename):
    return os.path.isfile(filename)

def process_pdf(filename):
    global embeddings_dict, unigram_probabilities, df_dict_vec, raw_df, raw_citation_context, df
    file_path = filename
    raw_text = functions.extract_raw_text_from_pdf(file_path)
    in_text_citations, cleaned_text = functions.extract_in_text_citations(raw_text)

    raw_citation_context = pd.DataFrame(columns=['citation', 'context'])
    for citation in in_text_citations:
        context = functions.get_context(raw_text, citation)
        new_row = pd.DataFrame({'citation': [citation], 'context': [context]})
        raw_citation_context = pd.concat([raw_citation_context, new_row], ignore_index=True)


    # extract Refernces part
    references = re.findall(r'References.*', cleaned_text) 
    # check if there is "Appendix" part after references
    appendix = re.findall(r'Appendix.*', references[0])
    # find the word "References" and remove everything after it
    cleaned_text = "title" + "author" + re.sub(r'References.*', '', cleaned_text) + ' '.join(appendix)

    all_sentences = cleaned_text.split(' ')
    chunks = [' '.join(all_sentences[i:i+250]) for i in range(0, len(all_sentences), 200)]

    # create a dataframe with the chunks
    raw_df = pd.DataFrame(chunks, columns=['text'])
    # process the sentences
    df = raw_df['text'].apply(functions.process_sentence)

    # create a vocabulary
    vocabulary = set()
    for sentence in df:
        for w in sentence:
            vocabulary.add(w)
    vocabulary = list(vocabulary)

    corpus = []
    for sentence in df:
        for word in sentence:
            corpus.append(word)

    # create a dictionary to store the unigram probability of each word
    unigram_probabilities = {}
    for word in vocabulary:
        unigram_probabilities[word] = functions.calculate_unigram_probability(corpus, word)

    # df_dict = df.to_dict() # set in correct input format
    # df_dict_vec = functions.sentence_embedding(embeddings_dict, df_dict, 0.5, unigram_probabilities)

    

def get_answer(question):
    global embeddings_dict, unigram_probabilities, df_dict_vec, raw_df, raw_citation_context, df
    
    in_test_query = functions.process_sentence(question)
    # remove "paper","text","article","thesis" from the query
    in_test_query = [x for x in in_test_query if x not in ["paper","text","article","thesis"]]

    # if the query contains 'reference','references' or 'citation' ,'citations'
    if any(x in in_test_query for x in ['reference','references','citation','citations']):
        df_chosen_temp = raw_citation_context
        # set the citation as the index of the dataframe
        df_chosen_temp = df_chosen_temp.set_index('citation')
        # df_chosen should be a dataframe series
        df_chosen = df_chosen_temp['context']
        # process the sentences
        df_chosen = df_chosen.apply(functions.process_sentence)
        test_query = in_test_query

    else:
        # find all the rows containing at least one word in the query
        selected_rows = []
        for i in range(len(df)):
            if any(x in df[i] for x in in_test_query):
                selected_rows.append(i)

        # check if the query if direct in the text or not
        if len(selected_rows) >= 1:
            # extract the selected_rows from the dataframe with thier index and create a new dataframe
            df_selected = df.iloc[selected_rows]
            df_chosen = df_selected 
            test_query = in_test_query
        else:
            df_chosen = df
            # expand query in some case
            expand_query = in_test_query.copy()
            for word in in_test_query:
                synonyms = functions.find_synonyms(word)
                try:
                    expand_query.append(synonyms.pop())
                except KeyError:
                    continue
            test_query = expand_query   

    df_dict = df_chosen.to_dict() # set in correct input format
    df_dict_vec = functions.sentence_embedding(embeddings_dict, df_dict, 0.5, unigram_probabilities)

    test_query_dict = {0: test_query}
    test_query_vec = functions.sentence_embedding(embeddings_dict, test_query_dict, 0.5, unigram_probabilities)

        
    # if the query contains 'reference','references' or 'citation' ,'citations'
    if any(x in in_test_query for x in ['reference','references','citation','citations']):

        # get top 5 similar chunks
        top_5_dict = functions.get_top5_similar_chunks(test_query_vec[0], df_dict_vec)
        top_5_dict

        # concanate all the key values of the top 5 chunks
        top_5_chunks = []
        for i in range(len(top_5_dict)):
            top_5_chunks.append(top_5_dict[i][0])

        #remove "()" in the in_text_citations
        in_text_citations0 = [re.sub(r'[()]', '', x) for x in top_5_chunks]
        # extract the citations contains ";" and split them into two citations
        in_text_citations1 = [x.split('; ') for x in in_text_citations0]
        # flatten the list
        in_text_citations2 = [item for sublist in in_text_citations1 for item in sublist]
        # create a doctionary to store the citations, the key is the citation and the value is the counts
        in_text_citations_dict = {}
        for i in in_text_citations2:
            if i in in_text_citations_dict:
                in_text_citations_dict[i] += 1
            else:
                in_text_citations_dict[i] = 1

        # order the dictionary by the counts
        in_text_citations_dict = dict(sorted(in_text_citations_dict.items(), key=lambda item: item[1], reverse=True))

        answer = "Answer: The top 5 important references are: \n (listed in order) \n" + str(list(in_text_citations_dict.keys())[:5])
        
    else:
        top_3_dict = functions.get_top3_similar_chunks(test_query_vec[0], df_dict_vec)

        # the the index from the dictionary keys
        top_3_index = [int(i[0]) for i in top_3_dict]
        final_chunk = ''
        for i in top_3_index:
            final_chunk += raw_df.iloc[i]["text"]
       
        answer = functions.ask_question(final_chunk, question)
    
    return "Computer:" + " " + answer + "\n"




# Welcome message
print("Computer: Which PDF would you like to ask a question about?(e.g. BERT.pdf)\n")

while True:
    # Prompt for the PDF filename
    filename = input("User: ")
    print("\n")

    if check_file_exists(filename):
        process_pdf(filename)
        print("File loaded successfully!\n")
        break
    if filename.lower() == "quit":
        print("Computer: Goodbye!\n")
        exit()
    else:
        print("Computer: File does not exist. Please enter a valid PDF filename.\n")

# Conversation loop
while True:
    # Ask for the user's question
    print("Computer: What is your question? (Enter 'quit' to exit).\n")

    user_question = input("User: ")
    print("\n")

    if user_question.lower() == "quit":
        print("Goodbye!")
        exit()

    answer = get_answer(user_question)
    print(answer + "\n")



