# Projet6-(Openclassrooms/CentraleSupelec)
Parcours Data Science

Projet n°6: "Classifiez automatiquement des biens de consommation"

## Description du projet

La marketplace e-commerce "Place de marche" souhaite étudier la faisabilité d'un moteur de classification des articles mis en place par les vendeurs. Pour l'instant, l'attribution de la catégorie d'un article est effectuée manuellement par les vendeurs et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit.

Pour rendre l’expérience utilisateur des vendeurs (faciliter la mise en ligne de nouveaux articles) et des acheteurs (faciliter la recherche de produits) la plus fluide possible et dans l'optique d'un passage à l'échelle, il devient nécessaire d'automatiser cette tâche.

Source des données : https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip

## Mission

Réaliser une première étude de faisabilité d'un moteur de classification d'articles, basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.
   - Manipulation de données textuelles, applications des techniques de NLP pour prétraitement des données textuelles : stopwords, lemmatisation / stemming, TF IDF / bag of words
   - Réduction de dimension par LDA / NMF
   - Prétraitement d'images et Extraction de features (SIFT)
   - Classification non supervisée
   - Construction d'un réseau de neurone
   - Transfer learning sur modèle imagenet
   - Mise en place d'une application de visualition dynamique

## Compétences évaluées

* Prétraiter des données image pour obtenir un jeu de données exploitable
* Représenter graphiquement des données à grandes dimensions
* Prétraiter des données texte pour obtenir un jeu de données exploitable
* Mettre en œuvre des techniques de réduction de dimension

## Livrables

* Un [notebook](https://github.com/raissaSaleu/P6_MBIADOU_SALEU/blob/main/P6_01_notebook.ipynb) contenant les fonctions permettant le prétraitement des données textes et images ainsi que les résultats du clustering (en y incluant des représentations graphiques).
* Un [support de présentation](https://github.com/raissaSaleu/P6_MBIADOU_SALEU/blob/main/Soutenance%20Projet%206.pdf) qui présente la démarche et les résultats du clustering.
