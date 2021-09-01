# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from resultat import *

app = Flask(__name__)

@app.route('/')
def formulaire():
    return render_template("formulaire.html")

@app.route('/resultats_tags.html', methods=['GET', 'POST'])
def resultats_tags():
    title=request.form['title']
    body=request.form['body']
    title_nettoye = nettoyage_text(title, 'title')
    body_nettoye = nettoyage_text(body, 'body')
    resultats_tags_Topic_global_LDA = resultat_prediction_Topic_global_LDA(body_nettoye, title_nettoye)
    resultats_tags_Topic_dominant_LDA = resultat_prediction_Topic_dominant_LDA(body_nettoye, title_nettoye)
    resultats_tags_classification = resultat_prediction_classification(body_nettoye, title_nettoye)
    return render_template("resultats_tags.html", body=body, title=title, resultats_tags_Topic_global_LDA=resultats_tags_Topic_global_LDA, resultats_tags_Topic_dominant_LDA=resultats_tags_Topic_dominant_LDA, resultats_tags_classification=resultats_tags_classification)

if __name__ == "__main__":
    app.run(debug=True)