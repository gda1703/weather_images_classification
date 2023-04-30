# weather_images_classification

Dans le cadre de la matière Deep Learning  enseigné par Mr Roman Yurchak au sein du Master 2 Modélisation statistiques économiques et financières, nous avons réalisé un modèle de classification d'images météorologiques à l'aide d'algorithmes de Deep Learning.  nous avons organisé le projet de la façon suivante : 
1. Un répertoire data contenant les sous répertoires suivants : 
    - Multi_class_ weather_dataset contenant les images selon leurs labels respectifs
    - Train contenant les données pour l’entraînement
    - Test contenant les données de test
    - Validation contenant les données de validation 

2. Un répertoire src contenant : 	
    - Le notebook nommé « DALLOMO_EDO_deep_Learning »  nécessaire à la réalisation du projet 
    - Un sous-répertoire nommé packages contenant :
      - L’utilitaire « utils_deep » contenant les fonctions utilisés au sein du notebook
      - Les différents modèles au format pickle pour faciliter leur réutilisation
3. Un répertoire dash_app contenant le code source de notre application web qui vous permettra d’avoir une vue d’ensemble du projet mais qui a également une fonctionnalité « Drag and Drop » qui vous permet à partir d’une image déposé par vos soins d’obtenir la prédiction du climat (ie levée de soleil, pluvieux, ensolleilé…) présent sur cette dernière.
Pour lancer l'application, vous devez vous positionner dans le répertoire dash_app et executer la commande qui suit : 
```
python app_DL.py
```

Merci d’avance à vous pour la lecture du projet.
