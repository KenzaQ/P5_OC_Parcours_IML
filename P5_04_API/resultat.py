#coding:utf-8
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import string
import re
import spacy
import sets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump, load
import json
nlp = spacy.load('en_core_web_sm')

## Récupération du DataFrame final

data_final = pd.read_csv("static/QueryResults_clean.csv", na_filter=False)

## Récupération des stopwords

sw_body = pd.read_csv("static/stopwords_body.csv")
sw_body = list(sw_body['sw_body'])
sw_title = pd.read_csv("static/stopwords_title.csv")
sw_title = list(sw_title['sw_title'])

## Nettoyage du message

def nettoyage_text(message, cas):

	## Enlever la ponctuation
	message = message.translate(str.maketrans('', '', string.punctuation))

	## Mettre les mots en minuscule
	message = message.encode().decode('utf-8').lower()

	## Conservation uniquement des noms
	sentence_nlp = []
	for word in nlp(message):
		if word.pos_ == 'NOUN':
			sentence_nlp.append(str(word))
			sentence_nlp.append(" ")
		elif word.pos_ == 'PROPN':
			sentence_nlp.append(str(word))
			sentence_nlp.append(" ")
		else:
			pass
	message = "".join(sentence_nlp)

	## Tokeniser
	tokens = message.split()

	## Suppression des stopwords
	if cas == 'body':
		sw = sw_body
	else:
		sw = sw_title
	sentence_stopwords = []
	for word in tokens:
		if word not in list(sw):
			sentence_stopwords.append(word)
			sentence_stopwords.append(" ")
		else:
			pass
	message_nettoye = "".join(sentence_stopwords)

	## Supprimer les nombres seuls
	message_nettoye = re.sub(" \d+", " ", message_nettoye)

	## Supprimer les mots à 1 lettre
	message_nettoye = message_nettoye.split()
	message_nettoye = [i for i in message_nettoye if len(i) > 1]
	sentence = []
	for word in message_nettoye:
		sentence.append(word)
		sentence.append(" ")
	message_nettoye = "".join(sentence)

	## Supprimer les espaces
	message_nettoye = " ".join(message_nettoye.split())

	return message_nettoye

## Récupération des liste des mots contenus dans tous les Topics

liste_mots_Topic_global_body = pd.read_csv("static/liste_mots_Topic_global_body.csv")
liste_mots_Topic_global_body = list(liste_mots_Topic_global_body['liste_globale_body'])
liste_mots_Topic_global_title = pd.read_csv("static/liste_mots_Topic_global_title.csv")
liste_mots_Topic_global_title = list(liste_mots_Topic_global_title['liste_globale_title'])

## Approche non supervisée : Tags créés avec le matching entre les mots contenus dans le message et les mots de tous les Topics

def resultat_prediction_Topic_global_LDA(body_nettoye, title_nettoye):

	## Match pour Body
	message_body = body_nettoye.split()
	liste_features_body = [i for i in message_body if i in liste_mots_Topic_global_body]
	liste_features_body = list(set(liste_features_body))

	## Match pour Title
	message_title = title_nettoye.split()
	liste_features_title = [i for i in message_title if i in liste_mots_Topic_global_title]
	liste_features_title = list(set(liste_features_title))

	return {'Body' : liste_features_body, 'Title' : liste_features_body}

## Récupération des dictionnaires du numéro du Topic associé à son Top 25 de mots

with open('static/dict_topics_body.json', 'r') as infile:
	dict_topics_body = json.load(infile)

with open('static/dict_topics_title.json', 'r') as infile:
	dict_topics_title = json.load(infile)

## Récupération des modèles LDA sélectionnés

best_lda_model_body = load('static/best_lda_model_body.joblib')
best_lda_model_title = load('static/best_lda_model_title.joblib')

## Approche non supervisée : Tags créés avec le matching entre les mots contenus dans le message et les mots du Topic dominant

def resultat_prediction_Topic_dominant_LDA(body_nettoye, title_nettoye):

	## Prédiction du Topic dominant pour Body via le modèle LDA
	tf_vectorizer = CountVectorizer(min_df = 2, max_features = 1000)
	data_vectorized = tf_vectorizer.fit_transform(data_final['Body_nettoye_stopwords'])
	body_lda = tf_vectorizer.transform([str(body_nettoye)])
	topic_probability_scores_body = best_lda_model_body.transform(body_lda)
	topic_dominant_body = np.argmax(topic_probability_scores_body)
	liste_mots_Topic_dominant_Body = dict_topics_body[str(topic_dominant_body)]

	## Match pour Body
	message_body = body_nettoye.split()
	liste_features_body = [i for i in message_body if i in liste_mots_Topic_dominant_Body]
	liste_features_body = list(set(liste_features_body))

	## Prédiction du Topic dominant pour Title via le modèle LDA
	tf_vectorizer = CountVectorizer(min_df = 2, max_features = 1000)
	data_vectorized = tf_vectorizer.fit_transform(data_final['Title_nettoye_stopwords'])
	title_lda = tf_vectorizer.transform([str(title_nettoye)])
	topic_probability_scores_title = best_lda_model_title.transform(title_lda)
	topic_dominant_title = np.argmax(topic_probability_scores_title)
	liste_mots_Topic_dominant_Title = dict_topics_title[str(topic_dominant_title)]

	## Match pour Title
	message_title = title_nettoye.split()
	liste_features_title = [i for i in message_title if i in liste_mots_Topic_dominant_Title]
	liste_features_title = list(set(liste_features_title))

	return {'Body' : liste_features_body, 'Title' : liste_features_body}

## Récupération des modèles de classification sélectionnés

modele_final_sup_body = load('static/modele_final_sup_body.joblib')
modele_final_sup_title = load('static/modele_final_sup_title.joblib')

## Approche supervisée : Tags Stack Overflow prédits à partir du vecteur de mots du message

def resultat_prediction_classification(body_nettoye, title_nettoye):

	data_final['Tags_nettoyes_liste'] = data_final.apply(lambda row : row['Tags_nettoyes'].split(', '), axis=1)
	target = pd.get_dummies(data_final['Tags_nettoyes_liste'].apply(pd.Series).stack()).sum(level=0)
	target = target.iloc[:, 1:]

	## Vectorisation pour Body
	tf_idf_vec = TfidfVectorizer()
	tf_idf_data = tf_idf_vec.fit_transform(data_final['Body_nettoye_stopwords'])
	tf_idf_body_nettoye = tf_idf_vec.transform([str(body_nettoye)])

	## Prédiction des Tags Stack Overflow via le modèle de classification sélectionné avec Body
	target_predict_body = modele_final_sup_body.predict(tf_idf_body_nettoye)
	dummies = pd.DataFrame(target_predict_body)
	dummies.loc[1] = target.columns
	dummies = dummies.transpose()
	dummies.columns = ['Presence', 'Tags']
	Tags_predict_body = list(dummies[dummies['Presence'] == 1]['Tags'])

	## Vectorisation pour Title
	tf_idf_vec = TfidfVectorizer()
	tf_idf_data = tf_idf_vec.fit_transform(data_final['Title_nettoye_stopwords'])
	tf_idf_title_nettoye = tf_idf_vec.transform([str(title_nettoye)])

	## Prédiction des Tags Stack Overflow via le modèle de classification sélectionné avec Title
	target_predict_title = modele_final_sup_title.predict(tf_idf_title_nettoye)
	dummies = pd.DataFrame(target_predict_title)
	dummies.loc[1] = target.columns
	dummies = dummies.transpose()
	dummies.columns = ['Presence', 'Tags']
	Tags_predict_title = list(dummies[dummies['Presence'] == 1]['Tags'])

	return {'Body' : Tags_predict_body, 'Title' : Tags_predict_title}
