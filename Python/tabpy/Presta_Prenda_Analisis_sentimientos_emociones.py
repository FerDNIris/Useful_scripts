#### TabPy Análisis de sentimientos de manera sencilla
import os 
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from unidecode import unidecode
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from scipy.special import softmax
import spacy

nlp = spacy.load('es_core_news_lg') 

lemmatizer = nlp.get_pipe("lemmatizer")

analyzer = SentimentIntensityAnalyzer()


classifier = pipeline('sentiment-analysis',
                      model="nlptown/bert-base-multilingual-uncased-sentiment")



def detectTextPolaritySentiment(text):
    try:
        nlpEval = nlp(text)
        return nlpEval._.blob.polarity
    except:
        return 0.0

def detectTextPolaritySentimentVader(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

#### Ahora vamos a usar BERT para el análisis de sentimiento generalista si fallan los dos anteriores

def categorizedRange(number):
    import re
    number = int(re.sub(r'\D', '', str(number)))
    return ((number -1)/4)*2-1
    
def bertSpanishClassifier(text):
    import numpy as np
    import re
    mainClassifier = classifier(text)
    resultsClassifier = mainClassifier[0]
    scoreClassifier = resultsClassifier['score']
    if scoreClassifier >= 0.75:
        return categorizedRange((resultsClassifier['label']))
    else:
        return None
    


def combinedSentimentAnalysis(text):
    """
    Combina tres funciones de análisis de sentimiento con jerarquía y maneja errores de BERT.
    """
    text = str(text)
    try:
        result1 = bertSpanishClassifier(text)
        if result1 is not None:
            return result1
    except Exception as e:
        print(f"Error en bertSpanishClassifier: {e}")
        pass

    result2 = detectTextPolaritySentimentVader(text)
    if result2 is not None:
        return result2

    result3 = detectTextPolaritySentiment(text)
    if result3 is not None:
        return result3

    return None   

def categorySentiment(sentimentScore):
    if sentimentScore>0:
        return 'Positive'
    elif sentimentScore<0:
        return 'Negative'
    else:
        return 'Neutral'
    

def detectEmotion(text):
  ### Get the pretrained model
  model_path = "daveni/twitter-xlm-roberta-emotion-es"
  tokenizer = AutoTokenizer.from_pretrained(model_path )
  config = AutoConfig.from_pretrained(model_path )
  emotions_model = AutoModelForSequenceClassification.from_pretrained(model_path)
  ### Starting the encoding
  text = str(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  try:
    output = emotions_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    emotions_score = np.sort(range(scores.shape[0]))
    emotions_score= emotions_score[0]
    l = config.id2label[ranking[emotions_score]]
    s = scores[ranking[emotions_score]]
    return l, np.round(float(s), 4)
  except:
    return None, None
  

def extract_sentiment_emotions(df):
    sentiment_score = []
    sentiment_classification = []
    emotion_label = []
    emotion_score = []
    for idx, text in df["input"].items():
        text = str(text)
        sentiment_score.append(combinedSentimentAnalysis(text))
        sentiment_classification.append(categorySentiment(combinedSentimentAnalysis(text)))
        emotion_label.append(detectEmotion(text)[0])
        emotion_score.append(detectEmotion(text)[1])
        print("Row " + str(idx) + " done...")
    df['sentiment_score'] = pd.Series(sentiment_score)
    df['sentiment_classification'] = pd.Series(sentiment_classification)
    df['emotion_label'] = pd.Series(emotion_label)
    df['emotion_score'] = pd.Series(emotion_score)
    return df

print('Pasó todo')

def get_output_schema():
    return pd.DataFrame({
        'respondent_id': prep_string(),
        'como_obtienes_dinero_necesitas': prep_string(),
        'razon_principal_eliges_solucion': prep_string(),
        'satisfaccion_solucion_actual': prep_string(),
        'por_que_satisfaccion_solucion_actual': prep_string(), 
        'sentiment_score': prep_decimal(),
        'sentiment_classification': prep_string(),
        'emotion_label': prep_string(),
        'emotion_score': prep_decimal()
    })