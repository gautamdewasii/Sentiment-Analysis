import numpy as np
import pandas as pd
import pickle
import streamlit as st
import re
import contractions
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# loading our model file (model.sav) into this program
l1_model=pickle.load(open('K:/[2]MY DOCS/Coding/jupyter notebook/projects/Sentiment Analysis/model.sav','rb'))


# loading tfidf vectorizer object file for text encoding 
cv_encoder=pickle.load(open('K:/[2]MY DOCS/Coding/jupyter notebook/projects/Sentiment Analysis/cv.sav','rb'))


def tweetCleaner(text):
    lemmatizer= WordNetLemmatizer()
    sklearn_stopwords=set(ENGLISH_STOP_WORDS)
    nltk_stopwords= set(stopwords.words("english"))
    combined_stopwords=nltk_stopwords.union(sklearn_stopwords)
    dataset_specific_stopword=['phone','io' , 'rt' , 'tweeter','follow','com']
    final_stopwords=dataset_specific_stopword + list(combined_stopwords)

    new_text=re.sub(r"'s\b", " is", text)
    new_text=re.sub("#","",new_text)
    new_text=re.sub(r"@[A-Za-z0-9]+","",new_text)
    new_text=re.sub(r"http\S+","",new_text)
    new_text=contractions.fix(new_text)
    new_text=re.sub(r"[^a-zA-Z]"," ",new_text)
    new_text=new_text.lower().strip()

    new_text= [token for token in new_text.split() if token not in final_stopwords]
    # removing words which are of 2 or less characters
    new_text= [token for token in new_text if len(token)>2 ]
    cleaned_text=''
    for token in new_text:
        cleaned_text=cleaned_text + lemmatizer.lemmatize(token) + ' '

    return cleaned_text
# main() for web app interface and input tasks
def main():
    
    # for wide look 
    st.set_page_config(layout="wide")


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/676578/pexels-photo-676578.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

    html_temp="""

    <div style="background-color:#e24150;padding:10xp">
    <h2 style="color:white;text-align:center;">😊 Sentiment Analysis Model 😡</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    input_tweet=str(st.text_area("**Enter any message :keyboard:**"))

    # 1) Tweet cleaning
    tweet=tweetCleaner(input_tweet)

    # 2) feature encoding 
    cv_encoded_tweet=cv_encoder.transform([tweet])

    # 3) prediction using model
    pred=l1_model.predict(cv_encoded_tweet)

    # 4) button for prediction
    if st.button("Predict"):
        if pred == 1:
            st.success("**Negative Tweet **:rage: ")
        else:
            st.success("**Positive Tweet **:blush:")


if __name__ == '__main__':
    main()
