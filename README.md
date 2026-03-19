# Logistic Regression Unittest and Logging

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jnooo-cloud/logistic-regression-unittest-logging/main?urlpath=lab/tree/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jnooo-cloud/logistic-regression-unittest-logging/blob/main/run_tests.ipynb)

Dieses Repository ist die Nachbearbeitung fuer Pruefungsaufgabe 2. Als Kursbeispiel wurde die Logistic-Regression-Aufgabe aus dem Udemy-Kurs "Python fuer Data Science, Machine Learning & Visualization" ausgewaehlt und um Logging, Timing und Python-Unittests erweitert.

## Inhalt

- `advertising_ml_system.py`: Python-Implementierung mit `my_logger`, `my_timer`, `train_model()`, `predict_with_model()` und Auswertungsfunktionen.
- `tests/test_advertising_model.py`: Python-Unittests fuer `predict()` und `fit()`.
- `data/Advertising.csv`: Originaldatensatz aus dem Kursbeispiel.
- `test_data/advertising_test_data.csv`: gespeicherte Testdaten fuer den Vorhersage-Testfall.
- `test_data/reference_runtime.json`: repraesentative Referenzlaufzeit fuer den Trainings-Testfall.
- `run_tests.ipynb`: Notebook zum Starten und Dokumentieren der Tests ueber Binder oder Colab.

## Ausfuehrung ueber Badge

### Binder

1. Klicke auf den Binder-Badge.
2. Oeffne `run_tests.ipynb`.
3. Fuehre alle Zellen aus.

### Colab

1. Klicke auf den Colab-Badge.
2. Fuehre die Zellen in `run_tests.ipynb` aus.

## Python-Unittests

Die beiden geforderten Testfaelle sind in `tests/test_advertising_model.py` implementiert:

1. `test_predict_function_on_saved_test_data`
   Dieser Test trainiert das Modell, laedt die Testdaten aus `test_data/advertising_test_data.csv`, fuehrt `predict()` aus und ueberprueft die Modellguete ueber Accuracy und Confusion Matrix.
2. `test_fit_runtime_is_within_120_percent_of_reference`
   Dieser Test misst mit `my_timer` die repraesentative Laufzeit wiederholter `fit()`-Aufrufe und prueft, dass die aktuelle Laufzeit nicht groesser als 120% der in `test_data/reference_runtime.json` abgelegten Referenzlaufzeit ist. Die wiederholten `fit()`-Aufrufe reduzieren Messrauschen in Binder- und Colab-Umgebungen.

## Lokaler Testaufruf

```bash
python -m unittest -v
```

## Erwartetes Ergebnis

Beim Testlauf werden beide Tests erfolgreich ausgefuehrt. Zusaetzlich wird in `logs/model_test.log` dokumentiert, wann die Funktionen gestartet wurden und wie lange der Laufzeit-Benchmark benoetigt hat.

## Dokumentierte Bildschirmausgabe

Die erfolgreiche Konsolenausgabe sieht wie folgt aus:

```text
test_fit_runtime_is_within_120_percent_of_reference (tests.test_advertising_model.TestAdvertisingModel.test_fit_runtime_is_within_120_percent_of_reference) ... ok
test_predict_function_on_saved_test_data (tests.test_advertising_model.TestAdvertisingModel.test_predict_function_on_saved_test_data) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.394s

OK
```
