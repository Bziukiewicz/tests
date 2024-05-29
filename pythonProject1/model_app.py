import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from scipy.io import loadmat
import numpy as np
import io
import base64
import scipy.io
import os
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import scipy.io
import csv
import ecg_plot
from dash.exceptions import PreventUpdate
import keras

import plotly.express as px
from plotly.subplots import make_subplots
import dash_table
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

from tensorflow.keras.models import load_model
# Załaduj zapisany model
model = load_model('moj_model.h5')

from joblib import Parallel, delayed
import joblib

# Load the model from the file
random_tree_model = joblib.load('moj_model_2.pkl')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='ekg-store'),
    dcc.Store(id='processed-ekg-store'),
    dcc.Store(id='button-states-store')
])

index_page = html.Div([
    html.Div([
        html.H3('Zespół Wolffa-Parkinsona-Whitea')
    ]),
    html.Div(['Zespół Wolffa-Parkinsona-Whitea  nazywany jest również zespołem preekscytacji. '
              'Charakteryzuje się on obecnością dodatkowej drogi przewodzenia między przedsionkami a komorami. ',
              html.Br(),
                'Zespół WPW jest wrodzoną wadą serca, dlatego diagnozowany jest u osób w bardzo różnym wieku. ',
                html.Br(),
                ' W mojej pracy skupiam się na anomaliach występujących na badaniu EKG. Charakterystyczne dla zespołu WPW są anoamlie wypunktowane poniżej. ',
                html.Br(),
              'Na stronie na której się znajdujesz możesz przyjrzeć się jak gołym okiem można zobaczyć pewne anomalie u pacjentów z zaburzeniem. ',
              'Możesz również sprawdzić jak przygotowane zostały dane do wprowadzenia ich do modelu, a także zobaczyć jaką skuteczność mają poszczególne modele. ',
              ' W ostatniej zakładce możesz zobaczyć od czego zależy występowanie zespołu WPW. ',
                html.Br(),
              'Zachęcam do przejścia do strony z predykcją, aby sprawdzić jak model działa na przesłanych danych.  ',
              dcc.Link('Przejdź do wykrywania zaburzenia na podstawie EKG', href='/predictions'),
              html.Hr()]),
    html.Div([
        html.Ul([
            html.Li('Anomalia 1: Skrócenie odstępu PQ'),
            html.Li('Anomalia 2: Obecność fali delta'),
            html.Li('Anomalia 3: Poszerzenie zespołu QRS'),
            html.Li('Anomalia 4: Zmiany odcinka ST-T')
        ]),
        html.Img(src='assets/SinusRhythmLabels.svg.png', title='źródło: Wikipedia.org' ,style={'margin-left': '20px', 'width':'500px', 'height'  : 'auto'})
    ], style={'display': 'flex', 'align-items': 'center'}),

    html.Div([
        dcc.Tabs([
            dcc.Tab(
                label='Porównanie EKG pacjentów zdrowych i z obecnością WPW',
                children=[
                    html.Div(['Na podstawie poniższych wyników badań można zauważyć, że u pacjenta z zespołem WPW poszerzony jest zespół QRS',
                              html.Br(),
                              'W oczy rzuca się również to, że odcinek RR jest skrócony, względem tego odcinka u zdrowego pacjenta. Oznacza to szybszy rytm serca. '
                              ' Jest to charakterystyczne przy częstoskurczach nadkomorowych. Dodatkowa droga przewodzenia może powodować przyśpieszony rytm serca, '
                              ' jednak nie jest to jedynie objawa WPW.',
                              html.Br(),
                              'Odcinek ST-T różni się względem zdrowego pacjenta.',
                              html.Br(),
                              'Bez zbliżenia nie jest łatwo dostrzec obecność fali delta. Obecność fali delta jest kluczowym kryterium przy diagnozie schorzenia WPW.'
                              ]),
                    html.Img(src='/assets/Unknown-2.png', style={'width': '100%', 'height': 'auto'}),
                    html.Img(src='/assets/Unknown.png', style={'width': '100%', 'height': 'auto'}),
                    html.Br(),

                ]
            ),
            dcc.Tab(
                label='Porównanie modeli uczenia maszynowego w skuteczności wykrywania WPW',
                children=[
                    dcc.Tabs([
                        dcc.Tab(label='Modele uczenia maszynowego',
                                children=[
                                    html.Div([
                                        html.H4("K-Nearest Neighbors (KNN)"),
                                        html.Iframe(
                                            src='assets/cm_knn.html',
                                            style={'width': '100%', 'height': '500px', 'border': 'none'}
                                        ),
                                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

                                    html.Div([
                                        html.H4("Regression Model"),
                                        html.Iframe(
                                            src='assets/cm_regression.html',
                                            style={'width': '100%', 'height': '500px', 'border': 'none'}
                                        ),
                                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

                                    html.Div([
                                        html.H4("Random Forest Model"),
                                        html.Iframe(
                                            src='assets/random_tree_cm.html',
                                            style={'width': '100%', 'height': '500px', 'border': 'none'}
                                        ),
                                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

                                    dash_table.DataTable(
                                        columns=[
                                                {'name': 'Model', 'id': 'model'},
                                                {'name': 'Accuracy score', 'id': 'acc'}
                                            ],
                                            data=[
                                                {'model': 'K-Nearest Neighbors (KNN)', 'acc': '87,5 %'},
                                                {'model': 'Regression Model',  'acc': '80 %'},
                                                {'model': 'Random Forest Model',  'acc': '91,25 %'}
                                            ],
                                        style_table={'width': '60%', 'margin': 'auto'},
                                        style_cell={'textAlign': 'center', 'fontSize': '16px'}

                                    )]),
                        dcc.Tab(label='Modele uczenia głębokiego',
                                children=[
                                        html.Div([
                                            html.H4("Convolutional Neural Network (cnn)"),
                                            html.Iframe(
                                                src='assets/cm_cnn.html',
                                                style={'width': '100%', 'height': '500px', 'border': 'none'}
                                            ),
                                            ],
                                            style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        html.Div([
                                            html.H4("Recurrent Neural Network (rnn)"),
                                            html.Iframe(
                                                src='assets/cm_rnn.html',
                                                style={'width': '100%', 'height': '500px', 'border': 'none'}
                                            ),
                                            ],
                                            style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        dash_table.DataTable(
                                            columns=[
                                                {'name': 'Model', 'id': 'model-1'},
                                                {'name': 'Accuracy score', 'id': 'acc-1'}
                                            ],
                                            data=[
                                                {'model-1': 'Convolutional Neural Network (cnn)', 'acc-1': '83,75 %'},
                                                {'model-1': 'Recurrent Neural Network (rnn)',  'acc-1': '70 %'}
                                            ],
                                            style_table={'width': '60%', 'margin': 'auto'},
                                            style_cell={'textAlign': 'center', 'fontSize': '16px'}
                                        )
                                ])
                    ])
                ]
            ),
            dcc.Tab(
                label='Przygotowanie danych EKG do detekcji WPW',
                children=[
                    dcc.Markdown(
                        """
                        ```
                        def resample_beats(beats):
                            return np.array([np.nan_to_num(signal.resample(np.asarray(i), 250)) for i in beats])

                        def median_beat(beat_dict):
                            beats = [entry['Signal'] for entry in beat_dict.values()]
                            rsmp_beats = resample_beats(beats)
                            return np.median(rsmp_beats, axis=0)

                        def process_ecgs(raw_ecg):
                            processed_ecgs = []
                            for lead_set in tqdm(raw_ecg):
                                twelve_leads = []
                                leadII = lead_set[1]
                                leadII_clean = nk.ecg_clean(leadII, sampling_rate=500, method="neurokit") 
                                r_peaks = nk.ecg_findpeaks(leadII_clean, sampling_rate=500, method="neurokit", show=False) 
                                for lead in lead_set:
                                    try:
                                        beats = nk.ecg_segment(lead, rpeaks=r_peaks['ECG_R_Peaks'], sampling_rate=500, show=False) 
                                        med_beat = median_beat(beats)  
                                    except:
                                        med_beat = np.ones(250) * np.nan
                                    twelve_leads.append(med_beat)
                                processed_ecgs.append(twelve_leads)
                            return np.array(processed_ecgs)
                        ```
                        """, style={'width': '50%', 'height': '1000px', 'border': 'none'}
                    ),
                    html.Img(src='/assets/Unknown-3.png', style={'width': '100%', 'height': 'auto'})
                ]
            ),
            dcc.Tab(
                label='Zależności',
                children=[
                    html.Iframe(
                        src='assets/wykres_rozkladu_plci.html',
                        style={'width': '50%', 'height': '600px', 'border': 'none'}
                    ),
                    html.Iframe(
                        src='assets/wykres_rozkladu_wieku.html',
                        style={'width': '50%', 'height': '600px', 'border': 'none'}
                    )
                ]
            )]
        )
    ])

])

prediction_page = html.Div([
    html.Div([
        html.H3('Przewidywanie Zespołu Wolfa-Parkinsona-Whitea'),
        html.Div('Poniżej można załadować plik EKG w formacie .mat, aby na jego podstawie uzyskać,'
                 ' predykcje obecności zespołu Wolfa Parkinsona Whitea.'
                 ' Przewidywanie odbywa się na podstawie dwóch modeli uczenia.'),
        html.Hr(),
        html.Div([
            dcc.Upload(
                id='upload-1',
                children=html.Div([
                    'Drag and drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'textAlign': 'center',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'lineHeight': '60px'
                },
                multiple=True
            )]),
        html.Div([
            html.Button('Dokonaj odszumiania danych', id='button-3',
                      style={'width': '550px', 'height': '50px', 'backgroundColor': 'blue', 'color': 'white', 'display': 'none'}),
            html.Br(),
            html.Div(id='button-3-div'),
            html.Br(),
            html.Button('Dokonaj predykcji za pomocą konwolucyjnych sieci neuronowych', id='button-2',
                      style={'width': '550px', 'height': '50px', 'backgroundColor': 'blue', 'color': 'white', 'display':'none'}),
            html.Br(),
            html.Div(id='button-2-div')]),

        html.Div(id='div-1')
    ]),
    dcc.Link('Powrót do strony głównej', href='/')
])

@app.callback(
    [Output('ekg-store', 'data'),
     Output('button-3', 'n_clicks'),
     Output('button-2', 'n_clicks')],
    [Input('upload-1', 'contents')],
    [State('upload-1', 'filename')]
)
def update_ekg_store(contents, filename):
    if contents is not None:
        try:
            mat_data = load_mat_files(contents, filename)
            # Pobieranie danych EKG z załadowanego pliku .mat
            # ekg_data = mat_data['val']
            ekg_data = get_ekg_data(mat_data)

            print('Dane EKG:', ekg_data)  # Drukuj dane EKG do konsoli w celu sprawdzenia

            return ekg_data,None,None # Zwróć dane do przechowania w komponencie dcc.Store
        except Exception as e:
            print(f'Błąd podczas przetwarzania danych EKG: {e}')
            return None, None, None
    return None, None, None


def get_ekg_data(mat_data):
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2:
            # Jeśli znajdziemy dwuwymiarową tablicę, zwracamy ją
            return mat_data[key]
    # Jeśli nie znajdziemy pasującej tablicy, zwracamy None
    return None


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/predictions':
        return prediction_page
    else:
        return index_page


def load_mat_files(contents,filename):
    if contents is not None:
        content_type, content_string = contents[0].split(',')
        decoded = base64.b64decode(content_string)

        # Tworzenie pliku tymczasowego na potrzeby odczytu danych z pliku .mat
        temp_file_path = 'temp.mat'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(decoded)

        # Odczyt danych z pliku .mat
        mat_data = loadmat(temp_file_path)

        # Usuwanie tymczasowego pliku
        os.remove(temp_file_path)
        return mat_data


@app.callback(
    Output('div-1', 'children'),
    [Input('upload-1', 'contents'),
     Input('ekg-store', 'data')],
    [State('upload-1', 'filename')]
)

#funkcja do wczytywania i wyswietlania zawartości pliku
def update_output(contents, filename,data):

    if contents is not None:
        try:
            mat_data=load_mat_files(contents, filename)
            # Pobieranie danych EKG z załadowanego pliku .mat
            #ekg_data = mat_data['val']
            ekg_data= get_ekg_data(mat_data)

            # Tworzenie subplotów dla każdego leadu
            fig = make_subplots(rows=6, cols=2, shared_xaxes=True, subplot_titles=[
                'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
            ])

            # Nazwy odprowadzeń
            leads_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

            # Dodawanie danych do subplotów i legendy
            for i, lead_name in enumerate(leads_names):
                row = (i // 2) + 1
                col = (i % 2) + 1
                fig.add_trace(go.Scatter(x=np.arange(5000), y=ekg_data[i, :], mode='lines', name=lead_name, line=dict(width=1, color='rgb(21, 56, 125)')), row=row, col=col)

            fig.update_layout(title='Dane EKG', yaxis_title='Amplituda', height=1200, width=1800,
                              plot_bgcolor='rgb(237, 206, 215)')

            # Dodanie legendy
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

            print(ekg_data.shape)
            return html.Div([
                dcc.Graph(figure=fig),
            ])
        except Exception as e:
            return html.Div([
                'Wystąpił błąd podczas przetwarzania pliku. Upewnij się, że plik jest w formacie .mat.', html.Br(),str(e)
        ])

import neurokit2 as nk  # Import neurokit2 as nk

#funkcja do próbkowania sygnału
#
def resample_beats(beats):
    # Funkcja resample_beats dokonuje resamplingu zbioru szczytów do długości 250 próbek
    return np.array([np.nan_to_num(signal.resample(np.asarray(i), 250)) for i in beats])

#funkcja do filtracji medianowej sygnału
def median_beat(beat_dict):
    # Funkcja median_beat oblicza medianę zbioru szczytów dla każdego słownika w zestawie
    beats = [entry['Signal'] for entry in beat_dict.values()]
    rsmp_beats = resample_beats(beats)
    return np.median(rsmp_beats, axis=0)


@app.callback(
    [Output('button-3-div', 'children'),
     Output('processed-ekg-store', 'data')],
    [Input('button-3', 'n_clicks'),
     Input('ekg-store', 'data')]
)
def process_ecgs(n_clicks, raw_ecg):
    if n_clicks is not None:
        if raw_ecg is None:
            return "Brak danych EKG do przetworzenia.", None

        try:
            raw_ecg = np.array(raw_ecg)  # Konwersja z powrotem do tablicy numpy
            processed_ecgs = []
            print(raw_ecg.shape)
            twelve_leads = []
            leadII = raw_ecg[1]
            leadII_clean = nk.ecg_clean(leadII, sampling_rate=500, method="neurokit")
            r_peaks = nk.ecg_findpeaks(leadII_clean, sampling_rate=500, method="neurokit", show=False)
            for lead in raw_ecg:
                try:
                    beats = nk.ecg_segment(lead, rpeaks=r_peaks['ECG_R_Peaks'], sampling_rate=500, show=False)
                    med_beat = median_beat(beats)
                except:
                    med_beat = np.ones(250) * np.nan
                twelve_leads.append(med_beat)
            print(np.array(twelve_leads).shape)
            print(np.array(twelve_leads))
            return "Dane EKG zostały przetworzone pomyślnie.", np.array(twelve_leads).tolist()
        except Exception as e:
            return f"Wystąpił błąd podczas przetwarzania danych EKG: {e}", None

    return "", None  # W przypadku gdy n_clicks jest None

@app.callback(
    Output('button-2', 'style'),
    [Input('button-3', 'n_clicks')]
)
def show_buttons(n_clicks):
    if n_clicks is not None:
        style= {'width': '550px', 'height': '50px', 'backgroundColor': 'blue', 'color': 'white', 'display': 'block'}
        return style
    else:
        style={'display': 'none'}
        return style

@app.callback(
    Output('button-3', 'style'),
    [Input('ekg-store', 'data')]
)
def show_button_3(data):
    if data:
        return {'width': '550px', 'height': '50px', 'backgroundColor': 'blue', 'color': 'white', 'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('button-2-div', 'children'),
    [Input('button-2', 'n_clicks'),
     Input('processed-ekg-store', 'data')]
)
def predict_cnn(n_clicks, processed_ekg):
    if n_clicks is not None and processed_ekg is not None:
        try:
            # Konwersja przetworzonych danych EKG na numpy array
            processed_ekg = np.array(processed_ekg)
            processed_ekg = np.moveaxis(processed_ekg, 1, 0)
            processed_ekg = np.expand_dims(processed_ekg, axis=0)

            # Wykonanie predykcji przy użyciu załadowanego modelu
            predictions = model.predict(processed_ekg)

            # Zakładamy, że model zwraca prawdopodobieństwo obecności choroby
            predicted_class = (predictions > 0.5).astype(int)

            if predicted_class[0] == 1:
                result = 'Predykcja: Obecność zespołu Wolfa-Parkinsona-Whitea'
            else:
                result = 'Predykcja: Brak zespołu Wolfa-Parkinsona-Whitea'

            return html.Div([html.H4(result)])
        except Exception as e:
            return html.Div([f'Wystąpił błąd podczas predykcji: {e}'])
    else:
        return None

if __name__ == '__main__':
    app.run_server(debug=True)
