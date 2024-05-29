import pytest
import dash
from dash.testing.application_runners import import_app
import base64

@pytest.fixture
def dash_app():
    # Importujemy aplikację Dash
    app = import_app("model_app")
    return app

def test_index_page(dash_duo, dash_app):
    # Uruchamiamy aplikację
    dash_duo.start_server(dash_app)

    # Sprawdzamy, czy tytuł strony głównej jest poprawny
    dash_duo.wait_for_text_to_equal("h3", "Zespół Wolffa-Parkinsona-Whitea", timeout=10)

    # Sprawdzamy, czy link do strony predykcji istnieje
    assert dash_duo.find_element("a[href='/predictions']")

def test_prediction_page(dash_duo, dash_app):
    # Przechodzimy do strony predykcji
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_text_to_equal("h3", "Zespół Wolffa-Parkinsona-Whitea", timeout=10)
    dash_duo.find_element("a[href='/predictions']").click()

    # Sprawdzamy, czy tytuł strony predykcji jest poprawny
    dash_duo.wait_for_text_to_equal("h3", "Przewidywanie Zespołu Wolfa-Parkinsona-Whitea", timeout=10)

    # Sprawdzamy, czy element upload istnieje
    assert dash_duo.find_element("#upload-1")

def test_upload_and_processing(dash_duo, dash_app):
    # Przechodzimy do strony predykcji
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_text_to_equal("h3", "Zespół Wolffa-Parkinsona-Whitea", timeout=10)
    dash_duo.find_element("a[href='/predictions']").click()

    # Sprawdzamy, czy element upload istnieje
    upload = dash_duo.find_element("#upload-1")

    # Symulujemy upload pliku .mat
    with open("tests/JS40776.mat", "rb") as f:
        content = f.read()
        encoded = base64.b64encode(content).decode()
        dash_duo.driver.execute_script(
            f"arguments[0].lastElementChild.value = '{encoded}'", upload
        )
        dash_duo.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", upload)

    # Sprawdzamy, czy pojawia się przycisk "Dokonaj odszumiania danych"
    dash_duo.wait_for_element("#button-3")

    # Klikamy przycisk "Dokonaj odszumiania danych"
    dash_duo.find_element("#button-3").click()

    # Sprawdzamy, czy pojawia się przycisk "Dokonaj predykcji za pomocą konwolucyjnych sieci neuronowych"
    dash_duo.wait_for_element("#button-2")

def test_prediction(dash_duo, dash_app):
    # Przechodzimy do strony predykcji
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_text_to_equal("h3", "Zespół Wolffa-Parkinsona-Whitea", timeout=10)
    dash_duo.find_element("a[href='/predictions']").click()

    # Symulujemy upload pliku .mat
    upload = dash_duo.find_element("#upload-1")
    with open("tests/test_data.mat", "rb") as f:
        content = f.read()
        encoded = base64.b64encode(content).decode()
        dash_duo.driver.execute_script(
            f"arguments[0].lastElementChild.value = '{encoded}'", upload
        )
        dash_duo.driver.execute_script("arguments[0].dispatchEvent(new Event('change'))", upload)

    # Klikamy przycisk "Dokonaj odszumiania danych"
    dash_duo.wait_for_element("#button-3").click()

    # Klikamy przycisk "Dokonaj predykcji za pomocą konwolucyjnych sieci neuronowych"
    dash_duo.wait_for_element("#button-2").click()

    # Sprawdzamy wynik predykcji
    dash_duo.wait_for_text_to_equal("h4", "Predykcja: Obecność zespołu Wolfa-Parkinsona-Whitea", timeout=10)
