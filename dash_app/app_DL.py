# -*- coding: utf-8 -*-


import base64
import math
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objs as go
import random
import seaborn as sns
import sys
import base64
import datetime
import io
from dash import dcc, html, dash_table
from base64 import b64encode
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image


from tensorflow.keras.applications.inception_v3 import InceptionV3


# Charger le modèle 
with open(r'../src/packages/inceptionV3_model.pkl', 'rb') as file:
    incv3 = pickle.load(file)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Creation de l'object qui permet de contenir l'application dash 
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

# Configuration du nom de l'application
app.title = 'Meteo_app'

# Codage et encodage de l'image
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

## Lecture de l'image qu'on va utiliser comme logo 
path_perf_lanet = 'lanet_perf.png'
pil_img = Image.open(path_perf_lanet)

path_perf_incp = 'inceptionV3_perf.png'
pil_img = Image.open(path_perf_incp)

path_perf_efn = 'efc_perf.png'
pil_img = Image.open(path_perf_efn)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Creation de l'object qui permet de contenir l'application dash 
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)


# Configuration du style des tabs personnalise
tabs_styles = {
    'height': '44px'}

tab_style = {
    'borderBottom': '2px solid black',
    'padding': '6px',
    'fontWeight': 'bold',
    'color': 'black'}

tab_selected_style = {
    'borderTop': '2px solid black',
    'borderBottom': '2px solid black',
    'backgroundColor': '#CECCD0',
    'color': 'black',
    'padding': '6px'}

                    
# Configuration du layout de l'application                                          
app.layout = html.Div(style={'fontFamily': 'Arial'},
    children=[
        html.Div(
            html.H1('Prédiction d’images météorologiques', style={'text-align': 'center', 'color':'black', 'margin-bottom': '20px'})
        ),
        dcc.Tabs(id="tabs-styled-with-inline",  value='tab-1', children=[
            dcc.Tab(label='Home page', value='tab-1', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Modèles et performances', value='tab-2', style=tab_style, selected_style=tab_selected_style), 
            dcc.Tab(label='Prédiction en temps réel', value='tab-3', style=tab_style, selected_style=tab_selected_style)
        ], style={'backgroundColor': 'lightgrey'}),
        html.Div(id='tabs-content-inline')
    ]
)

# Configuration des tabs
@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))

def render_content(tab):    
    if tab == 'tab-1':
       return html.Div([
           html.H4("Cadre du projet", style={'text-align': 'center', 'margin-top':'30px'}),
    html.Div([ 
    html.P([
    "Dans le cadre du projet de ma matière Deep Learning enseigné par Monsieur R.Yurchak, nous avons réalisé un modèle de classification  d'images météorologiques. En effet, sur la base de données trouvées sur le site ", 
    html.A("https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset", href="https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset"), 
    ", le problème qui nous a été soumis était de trouver le climat présent sur une image donnée. ", 
    html.Br(),
    html.Br(),
    "La base de données contenait des labels associées à 4 climats, respectivement: ",
html.Ul([
        html.Li(" ☁️ nuageux"),
        html.Li(" 🌧️ pluvieux"),
        html.Li(" ☀️ ensoleillé"),
        html.Li(" 🌅 levée du soleil"),]),
    html.Br(),
    "Ainsi, nous avons implémenté différents modèles de classification d'images que vous pourrez trouver dans les sections suivantes de notre interface web  après avoir effectuer un preprocessing propres aux données d'images.",
])
        ], style={'padding': '20px', 'justifyContent': 'space-between', 'border': '3px solid #CECCD0', 'margin-bottom':'150px', 'margin-top':'50px'}), 
    
    html.Div([
        html.H4("CONTACTS 👨‍💻", style={'text-align': 'center', 'margin-bottom': '20px', 'color': 'black'})
    ], style={'margin': 'auto', 'background-color': '#CECCD0',  'width': '100%', 'height': '40px', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top':'100px'}),
    html.Div([                    
        html.Div([
            html.P(html.Strong("Giulia Dall'Omo"), style={'text-align': 'center', 'color': 'black'}),
            html.P("MoSEF - Stellantis"),
            html.P([html.Span("Git: "), html.A("Click here", href="https://github.com/gda1703/", target="_blank")]),
            html.P([html.Span("Linkedin: "), html.A("Click here", href="https://www.linkedin.com/in/giulia-dallomo/", target="_blank")])
        ], style={'border-right': '2px solid black', 'padding-right': '50px', 'margin-right': '30px', 'margin-left': '20px'}),
        html.Div([
            html.P(html.Strong("Amande Edo"), style={'text-align': 'center', 'color': 'black'}),
            html.P("MoSEF - BRED"),
            html.P([html.Span("Git: "), html.A("Click here", href="https://github.com/amandeedo/", target="_blank")]),
            html.P([html.Span("Linkedin: "), html.A("Click here", href="https://www.linkedin.com/in/amande-edo-2b51051b3/", target="_blank")])
        ], style={'padding-left': '20px', 'margin-left': '20px'}),
    ], style={'margin': 'auto', 'background-color': '#CECCD0', 'width': '100%', 'height': '150px', 'display': 'flex', 'justify-content': 'center', 'text-align': 'center', 'align-items': 'center'})
])

        
    # Creation de la page qui contient les modèles
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Modèles et performances', style={'text-align': 'center'}),
            html.Label('Sélectionner un modèle'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'LeNet', 'value': 'model1'},
                    {'label': 'InceptionV3', 'value': 'model2'}, 
                    {'label': 'EfficientNet', 'value': 'model3'}
                ],
                value='model1'
            ), 
            
            html.Div(id='model-metrics')
        ])
    
    # Creation de la page prédiction en temps réel
    elif tab == 'tab-3':
        return html.Div([
            html.H4("Prédictions en temps réel", style={'text-align': 'center'}), 
        html.Div([
    html.P([
    "Notre application Web comprend une fonctionnalité puissante qui permet aux utilisateurs de glisser-déposer une image sur l'interface, qui est ensuite automatiquement traitée par notre modèle d'apprentissage en profondeur. Ce modèle est basé sur l'architecture InceptionV3, qui est largement considérée comme l'un des modèles de reconnaissance d'images les plus puissants disponibles aujourd'hui.",
    html.Br(),
    html.Br(),
    "En utilisant cette fonctionnalité de glisser-déposer, les utilisateurs peuvent rapidement et facilement télécharger des images et obtenir des prédictions à partir de notre modèle. Il s'agit d'un outil extrêmement précieux pour quiconque a besoin de classer ou d'identifier des images de manière rapide et efficace. Que vous soyez un météorologue cherchant à identifier différents types de modèles météorologiques ou un chercheur qui doit analyser de grandes quantités de données, cette fonctionnalité peut vous aider à obtenir les résultats dont vous avez besoin rapidement et facilement.",
])
    
        ], style={'padding': '20px', 'justifyContent': 'space-between', 'border': '3px solid #CECCD0', 'margin-top':'50px'}), 
        
        html.Div([
     
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop ou ',
                    html.A('Selectionner une image')
                ]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
        ]),
        html.Div(id='output-image-upload'),
        html.Div(id='output-prediction'), 
    ])




# Configuration de la page des modeles 
@app.callback(
    Output('model-metrics', 'children'),
    Input('model-dropdown', 'value'))
    
def update_model_metrics(model):
    if model == 'model1':
        # Metriques du modele 
        return html.Div([
    html.H3("LaNet", style={'color': 'black', 'text-align': 'center'}),
    
    
    html.Div([
    html.P([
    "L’architecture ",
    html.Strong("LeNet"),
    " a été introduite par Yann LeCun en Novembre 1998 dans le journal Proceedings of the IEEE. C'est l'une des premières et plus simples architectures de réseau de neurones convolutifs (CNN). Au départ, elle a été construite pour la reconnaissance de caratères manuscrits et a ouvert la voie à des nombreuses autres architectures de CNN plus complexes. Elle est donc devenue le point de départ pour la classification d'images. ",
    html.Br(),
    html.Br(),
    "L'architecture du LeNet est constituée des couches suivantes:",
    html.Ul([
        html.Li("Convolution 1 : 30 filtres, dimension d'entrée (28, 28, 1), dimension du noyau (5, 5), fonction d'activation ReLU, pas de dépassement du noyau. Dans notre cas, les images sont de la taille 224x224 donc nous avons changé la dimension d'entrée pour notre problème."),
        html.Li("Max-Pooling 1 : dimension du pooling (2, 2)."),
        html.Li("Convolution 2 : 16 filtres, dimension du noyau (3, 3), fonction d'activation ReLU, Pas de dépassement du noyau."),
        html.Li("Max-Pooling 2 : dimension du pooling (2, 2)."),
        html.Li("Dropout : Connexions coupées: 20%. (que nous avons mis à 40%)"),
        html.Li("Aplatissement"),
        html.Li("Dense 1 : 128 neurones, fonction d'activation ReLU."),
        html.Li("Dense 2 : 10 neurones, fonction d'activation softmax (que nous avons mis à 4 puisque nous avons 4 labels)"),
    ]),
    html.Br(),
    "L'architecture ",
    html.Strong("LeNet est une des plus simples"),
    " à implémenter et une des plus importantes en Deep Learning car elle est le pilier de nombreux modèles plus complexe."
])

    
        ], style={'padding': '20px', 'justifyContent': 'space-between', 'border': '3px solid #CECCD0'}), 

    
    html.H4('Métriques', style={'color': 'black', 'text-align': 'center'}),
    html.Div([
        html.Div([
            html.H5("Train Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.3874')
        ], style={'padding': '30px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Train Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.8633')
        ], style={'padding': '30px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Validation Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.2990')
        ], style={'padding': '30px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}), 
        html.Div([
            html.H5("Validation Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.9107')
        ], style={'padding': '30px', 'text-align': 'center', 'border': '3px solid #CECCD0'}), 
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px', 'padding-top': '20px', 'margin-bottom': '50px'}),
    
    # Graphiques cote à cote
    html.H4('Evolution de la performance du modèle', style={'color': 'black', 'text-align': 'center'}),
    html.Div([
    html.Img(src=b64_image(path_perf_lanet), width=1000)], style={'text-align': 'center','border': '3px solid #CECCD0', 'padding': '10px'}), 
        
        ])
             

    
    elif model == "model2":
        # Metriques du modele 
        return html.Div([
    html.H3("InceptionV3", style={'color': 'black', 'text-align': 'center'}),
    
    
    html.Div([
    html.P([
    html.Strong("InceptionV3"),
    " est un modèle de réseau de neurones profond pour la classification d'images mais aussi pour les véhicules autonomes et l'imagerie médicale et il a été développé par des chercheurs de Google et a été publié en 2015 en tant qu'amélioration par rapport à son prédécesseur, InceptionV1.",
    html.Br(),
    html.Br(),
    "Le modèle InceptionV3 utilise une architecture de réseau de neurones convolutif avec plusieurs couches, y compris des couches convolutives, des couches de regroupement et des couches entièrement connectées. L'une de ses principales caractéristiques est l'utilisation de modules de démarrage, qui sont conçus pour capturer efficacement des caractéristiques à différentes échelles et résolutions. Le modèle est formé sur un grand ensemble de données d'images labellisées et apprend à reconnaître les modèles et les caractéristiques dans les images. Une fois formé, le modèle peut être utilisé pour classer de nouvelles images dans différentes catégories ou détecter des objets dans des images.",
	html.Br(),
	html.Br(),
	html.Strong("InceptionV3 "),
	"a atteint des performances de pointe sur une gamme d'ensembles de données de classification d'images de référence, tels qu'ImageNet.", 
	html.Br(),
	html.Br(),
	"Dans un premier temps, nous avons chargé le modèle InceptionV3 pre-entrainé qui a été formé sur le jeu de données ImageNet. Les trois paramentrès que nous avons sont :",

    html.Ul([
        html.Li("weights spécifie que les poids pré-formés doivent être chargés."),
        html.Li("include_top = False pour exclure la dernière couche entièrement connectée du modèle, ce qui nous permet d'ajouter nos propres couches de classification. "),
        html.Li("input_shape spécifie la taille des images d'entrée attendues par le modèle."),
    ]),
    html.Br(),
    "Ensuite, nous avons crée un modèle sequentiel auquel nous avons ajouté le modèle InceptionV3 en tant que première couche. Une couche", 
    html.Strong(" GlobalAveragePooling2D "), "est ensuite ajoutée pour réduire les dimensions spatiales de la sortie du modèle pré-entraîné.  Puis une couche", 
    html.Strong(" Dense "), "entièrement connectée avec 250 unités et une fonction d'activation ReLU, une couche", 
    html.Strong(" Dropout "), "pour régulariser le réseau mise à 0.4 (cela signifie que le 40% les connextion entre les couche précédente et la couche suivant ont été coupées), et une couche Dense finale avec 4 unités qui correspondent au nombre des classes d'output avec une fonction d'activation softmax pour renvoyer des prédictions de probabilité pour chaque classe.", 
    html.Br(),
    html.Br(),
"Pour la compilation du modèle, nous avons utilisé l'optimiseur", 
html.Strong(" Adam "), "avec un taux d'apprentissage de 0,001 et la fonction de perte d'entropie croisée catégorielle est utilisée pour former le modèle. ", 
    html.Br(),
    html.Br(),
"Concernant l'entrainement du modèle, le modèle a été entrainé sur les données d'entraînement (train_generator) et validé sur les données de validation (validation_generator) pour un maximum de 25 époques. Un rappel", 
html.Strong(" EarlyStopping "),  "est également utilisé pour arrêter l'entraînement plus tôt si la perte de validation ne s'améliore pas pendant 3 époques consécutives. Cela permet d'optimiser le taux d'apprentissage du modèle. ", 
])

    
        ], style={'padding': '20px', 'justifyContent': 'space-between', 'border': '3px solid #CECCD0'}),
    

    
    html.H4('Métriques', style={'color': 'black', 'text-align': 'center'}),
    html.Div([
        html.Div([
            html.H5("Train Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.1120')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Train Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.9600')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Validation Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.1869')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}), 
        html.Div([
            html.H5("Validation Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.9464')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px', 'padding-top': '20px', 'margin-bottom': '50px'}),
    #Graphiques cote à cote
    html.H4('Evolution de la performance du modèle', style={'color': 'black', 'text-align': 'center'}),
    html.Div([
    html.Img(src=b64_image(path_perf_incp), width=1000)], style={'text-align': 'center','border': '3px solid #CECCD0', 'padding': '10px'}),
    
])
    

    elif model == "model3":
        # Metriques du modele 
        return html.Div([
    html.H3("EfficientNet", style={'color': 'black', 'text-align': 'center'}),
    
    
    html.Div([
    html.P([
    html.Strong("EfficientNet "),
    "est une architecture de réseau de neurones profonds qui a été développée en 2019. Ce modèle fait partie des modèles du", 
    html.Strong(" state-of-the-art "), "des performances sur les différentes tâches de Computer vVision.",
    html.Br(),
    html.Br(),
    "L'architecture d'EfficientNet est basée sur une méthode de mise à l'échelle composée qui permet au modèle d'équilibrer sa profondeur, sa largeur et sa résolution à chaque couche. Plus précisément, le modèle est composé de plusieurs blocs(qui contient une série de couches convolutives, de normalisation par lots et de fonctions d'activation).",
	html.Br(),
	html.Br(),
	"Le bloc de construction principal de l'architecture EfficientNet est la convolution de goulot d'étranglement inversé mobile (MBConv). Le bloc MBConv se compose de trois composants principaux :",
    html.Ul([
        html.Li("Convolution en profondeur : il s'agit d'une opération de convolution légère qui applique un seul filtre à chaque canal d'entrée séparément. Elle est suivie d'une normalisation et d'une activation par lots."),
        html.Li("Goulot d'étranglement inversé : il s'agit d'une fonction non linéaire qui augmente le nombre de canaux, puis applique une convolution ponctuelle pour réduire la dimensionnalité."),
        html.Li("Goulot d'étranglement linéaire : il s'agit d'une combinaison de fonctions de convolution ponctuelle, de normalisation par lots et d'activation qui réduisent le nombre de canaux à la valeur d'origine."),
    ]),
    html.Br(),
    "L'architecture EfficientNet comprend également une technique appelée", 
    html.Strong(" activation swish "), "qui est une fonction d'activation non linéaire qui est plus efficace en termes de calcul que l'activation ReLU.", 
html.Br(),
html.Br(),
"Enfin, EfficientNet utilise une technique appelée", 
html.Strong(" mise à l'échelle composée "),  "qui consiste à mettre à l'échelle la profondeur, la largeur et la résolution du modèle de manière équilibrée. Cela permet au modèle d'atteindre une grande précision sur un large éventail de tâches tout en étant efficace en termes de calcul.", 
html.Br(),
html.Br(),
"Concernant la construction de l'architecture, nous avons procedé comme pour le modèle InceptionV3.", 
])

    
        ], style={'padding': '20px', 'justifyContent': 'space-between', 'border': '3px solid #CECCD0'}),
        
    html.H4('Métriques', style={'color': '#1A2F5C', 'text-align': 'center'}),
    html.Div([
        html.Div([
            html.H5("Train Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.1423')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Train Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.9556')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}),
        html.Div([
            html.H5("Validation Loss", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.2002')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0', 'margin-right': '20px'}), 
        html.Div([
            html.H5("Validation Accuracy", style={'color': 'black', 'text-align': 'center'}),
            html.Hr(style={'border-color': '#CECCD0', 'width': '50%'}),
            html.P('0.9464')
        ], style={'padding': '20px', 'text-align': 'center', 'border': '3px solid #CECCD0'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px', 'padding-top': '20px', 'margin-bottom': '50px'}),
    # Graphique 
    html.H4('Evolution de la performance du modèle', style={'color': 'black', 'text-align': 'center'}),
    html.Div([
    html.Img(src=b64_image(path_perf_efn), width=1000)], style={'text-align': 'center','border': '3px solid #CECCD0', 'padding': '10px'}),
])



def prediction(img):
    img = img.resize((224, 224)) 
    test_array = np.array(img)
    test_array= test_array/255.0
    test_array = np.expand_dims(test_array, axis=0)

    # Prédictions 
    predicted_probabilities = incv3.predict(test_array)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    
    labels = ['Nuageux ☁️ ', 'Pluvieux 🌧️ ', 'Ensoleillé ☀️ ', 'Levée du soleil 🌅 '] 
    
    return labels[predicted_labels[0]]


image_data = None

# Definition de la fonction qui parse l'image chargée 
def parse_contents(contents, filename, date):
    global image_data
    image_data = contents.encode('utf-8').split(b';base64,')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Récuperer les labels prédits
    predicted_label = prediction(image)
    
    return html.Div([
        html.H6("Le nom de l'image que vous avez choisit de prédire est : " +filename),
        html.Img(src=contents),
        html.Hr(),
        html.H4(f"Classe prédite : {predicted_label}")
    ])

# Définition du callback qui permet de mettre à jour l'output 
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
    

# Fin de l'application
if __name__ == "__main__":
    app.run_server(debug=True)

