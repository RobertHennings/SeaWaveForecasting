#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 21:18:04 2020

@author: Robert_Hennings
"""

#Begleitcode für das Projekt Meereswellenforecasting und das Berechnen weiterer Wellenparameter
#Ziel ist es die Parameter Wellenhöhe, Fetchlänge und Wellenperiode zu bestimmen

#Nötige Inputdaten sind:
#Windgeschwindigkeit und Richtung
#Dauer der Windgeschwindigkeit
#Temperatur der Luft
#Temperatur Wasser
#Messhöhe der Station
#Standort der Messstation (Land/Wasser)


# Import Meteostat library und dependencies sowie weiterer libraries
from datetime import datetime
from meteostat import Daily
from meteostat import Hourly
from meteostat import Stations
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# Eingabe der Standortdaten, hier Laboe in Schleswig-Holstein
lat = 54.40949
lon = 10.22698

# Zeitfenster deklarieren für das die Wetterdaten ermittelt werden sollen
start = datetime(1991, 2, 2) 
end = datetime(2020, 12, 25) 

# Get closest weather station to Laboe
#Resulatat ist der standort nördlich von Eckenrförde: Olpenitz 
stations = Stations()
stations = stations.nearby(lat, lon)
stations = stations.inventory('daily', (start, end))
station = stations.fetch(1)

# Stündlicher Datenimport für den gewählten Standort und das Zeitfenster
data = Hourly(station, start, end)
data = data.fetch()


#Zwischensicherung des Grunddatenstamms in den Dateien

data = pd.read_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Stündliche Grunddaten/GrunddatenStündlicheDaten.csv", index_col = 0)

data.describe()
#data.drop(data.iloc[:, 7:22], inplace = True, axis = 1)
# Plot line chart including average, minimum and maximum temperature
#data.plot(y=['tavg', 'tmin', 'tmax'])

#Plotten ausgewählter Daten
data.plot(y=['temp', 'wspd', 'dwpt'])
plt.xticks(rotation=45)
plt.show()
#Plotten der Temperatur
data.plot(y=['temp'])
plt.xticks(rotation=45)
plt.show()

print(len(data.temp))
data.describe()
data.info()
#260637 Zeilen->Beobachtungen


#Wetterdaten die hier vorliegen:
#Durchschnittstemperatur des Stunde
#Windgeschwindigkeit in km/h in jeder Stunde
#Windrichtung (wdir) in jeder Stunde
#Peak Wind Gust in km/h in keder Stunde
#Sunshine in total minutes tsun in jeder Stunde
#...

#Was noch an Daten benötigt wird:
#Wassertemperatur 
#Workaround: Ein Jahr nehmen pro Monat die Durchschnittswassertemperatur nehmen

#Wassertemperatur nun auch drin als Durcschnittswerte pro Monat in Kiel 



#Hinzufügen des Jahres als Spalte
data['Jahr'] = pd.DatetimeIndex(data.index).year
data['Monat'] = pd.DatetimeIndex(data.index).month
conditions1 = [
    (data['Monat'] == 1),
    (data['Monat'] == 2),
    (data['Monat'] == 3),
    (data['Monat'] == 4),
    (data['Monat'] == 5),
    (data['Monat'] == 6),
    (data['Monat'] == 7),
    (data['Monat'] == 8),
    (data['Monat'] == 9),
    (data['Monat'] == 10),
    (data['Monat'] == 11),
    (data['Monat'] == 12)
    ]

values1 = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August','September', 'Oktober', 'November','Dezember']

#Hinzufügen des Monats als Zahl und als Wort je Reihe
data['MonateAlsWort'] = np.select(conditions1, values1)

#Setzen der Meerestemperatur pro Monat als Durchschnittswert
valuesTemp = [3,2,3,6,11,15,18,18,16,13,9,6]

#Hinzufügen als Variable im Datensatz
data['SeaTemperature'] = np.select(conditions1, valuesTemp)

#Umrechnung der Km/h in m/s
data['GeschwInM/S'] = data['wspd']/3.6 

#Berechnung der Luft Wsser Temperaturdifferenz
data['LuftWasserDiff'] = data['temp']-data['SeaTemperature'] 

#data = pd.DataFrame(data)
#Letzte benötigte DAten angeben die konstant bleiben

WindgeschwindigkeitLand = pd.DataFrame(data['GeschwInM/S'])
 #m/s
#Dauer der Windstärke: 1h 20min, in Minuten: 72
Einwirkzeit = 60 #in Minuten sind hier 1,20h 72 wird dann im folgenden aug 60 gesetzt da für jede h ein Wert geschätzt wird
#Windrichtung: Südostwind
Messhöhe = 2 #m über Land
Lufttemperatur = data['temp'] #Grad Celcius
Wassertemperatur = data['SeaTemperature']
#Erdbeschleunigung
g = 9.81
#Berechnung einzelner wichtiger Parameter und Werte zur Umrechnung:
data['WindgeschwInMesshöhe'] = data['GeschwInM/S']*((10/Messhöhe)**(1/7))


#Umrechnung der Windgeschwindigkeit über Land zu über Wasser da dieser dort schneller ist
          
#Anlegen der Umrechnungsfaktoren als Variable für Berechnung der Windgeschwindigkeit über WAsser mit Temperaturdifferenz
conditions5 = [
    (data['LuftWasserDiff'] == 0),
    ((data['LuftWasserDiff'] > 0) & (data['LuftWasserDiff']<= 5)),
    ((data['LuftWasserDiff'] <= 10) & (data['LuftWasserDiff']> 5)),
    ((data['LuftWasserDiff'] <= 15) & (data['LuftWasserDiff']> 10)),
    ((data['LuftWasserDiff'] <= 20) & (data['LuftWasserDiff'] >15)),
    (data['LuftWasserDiff'] > 20),
    ((data['LuftWasserDiff'] <0) & (data['LuftWasserDiff']>= -5)),
    ((data['LuftWasserDiff'] < -5) & (data['LuftWasserDiff']>= -10)),
    ((data['LuftWasserDiff'] < 10) & (data['LuftWasserDiff']>= -15)),
    ((data['LuftWasserDiff'] < 15) & (data['LuftWasserDiff']>= -20)),
    (data['LuftWasserDiff'] < 20)
    ]

values5 = [1,0.92,0.85,0.8,0.78,0.77,1.05,1.15,1.18,1.21,1.23]
data['UmrechnungWindWassermitLuftDiff'] = np.select(conditions5,values5)

conditions3 = [
    (data['GeschwInM/S'] <18.5),
    (data['GeschwInM/S'] >= 18.5)]
values3 = [data['GeschwInM/S'],18.3]

data['InputWindUmrechnung'] = np.select(conditions3,values3) #als Input deklariert

WindgeschwindigkeitLand['FertigerFormelInput'] = data['InputWindUmrechnung']
WindgeschwindigkeitLand['Neu'] = 2.3-1.10488*np.log10(WindgeschwindigkeitLand['FertigerFormelInput'])

#InputwindUmrechnung muss zu float converted werden
data['InputWindUmrechnung'].dtypes

WindgeschwindigkeitLand.reset_index(drop =True,inplace=True)


#Zurücksetzten des Index 
#data.reset_index(drop=True, inplace=True)

#Umrechnungsfaktor der noch auf die Windgeschwindigkeit angewendet werden muss
data['WindüberWasserFaktor'] = 2.3-1.10488*np.log10(data['InputWindUmrechnung'])
#Windgeschwindigkeit über Wasser aber noch ihne Einberechnung der Luft-Wasser Temperaturdifferenz
data['WindgeschwüberWasser'] = data['WindgeschwInMesshöhe']*data['WindüberWasserFaktor']




#UmrechnungWind(WindgeschwindigkeitLand)
LuftWasserDifferenz = data['LuftWasserDiff']

#Umrechnung der WindgeschwindigkeitWasser unter Beachtung der Temperaturdifferenz


data['WindgeschwFertig'] = data['WindgeschwüberWasser']*data['UmrechnungWindWassermitLuftDiff']



#Berechnung Windspannungsfaktor nach Resio und Vincent

data['Windspannungsfaktor'] = ((0.53+0.047*data['WindgeschwFertig'])**0.5)*data['WindgeschwFertig']


#Berechnung des Fetches (Einwirkweg der Windgeschwindigkeit)
#Dimensionsloser Fetch zunächst

#Zeitumrechnung in Sekunden:
ZeitinSekunden = Einwirkzeit *60

data['FetchDimlos'] = (1.752*(10)**-3)*(((g*ZeitinSekunden)/data['Windspannungsfaktor'])**(3/2))

#Fetch in m berechnen

data['FetchinM'] = ((data['FetchDimlos']*(data['Windspannungsfaktor'])**2)/g)
#print("Der Fetch in m im Tiefwasser lautet:", FetchinM)

#Wellenhöhe berchnen:

data['Wellenhöhe'] = 0.0016*(((g*data['FetchinM'])/((data['Windspannungsfaktor'])**2))**0.5)*(((data['Windspannungsfaktor'])**2)/g)

#Wellenperiode berechnen:
data['Wellenperiode'] = 0.2857*(((g*data['FetchinM'])/((data['Windspannungsfaktor'])**2))**(1/3))*(data['Windspannungsfaktor']/g)




data.describe()

data.plot(y=['Wellenhöhe'])
plt.xticks(rotation=45)
plt.show()

data.plot(y=['SeaTemperature'])
plt.xticks(rotation=45)
plt.show()



data = pd.read_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Fertige Daten mit allen Variablen/FertigesModellAlleParameter.csv")
data.describe()
data.info()
#Vergleich der mathematischen Formeln mit ML Algorithmen zur Regression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = data.dropna(axis = 0, how = 'any')
data.info()

data.reset_index(drop=True, inplace=True)

data.drop('time', axis = 1, inplace = True)
data.drop('Unnamed: 0', axis = 1, inplace = True)

y = data['Wellenhöhe']
data.drop('Wellenhöhe', axis = 1, inplace = True)
data.drop('MonateAlsWort', axis = 1, inplace = True)
data.drop('prcp', axis = 1, inplace = True)
data.drop('coco', axis = 1, inplace = True)
data.drop('snow', axis = 1, inplace = True)
data.drop('wpgt', axis = 1, inplace = True)

data['tsun'].fillna(data['tsun'].mean(),inplace = True)
print(data['tsun'].mean())
X = data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=47)

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')

data.isna()
regressor.fit(X,y)

y_pred = regressor.predict(X_test)


#MAE
print(metrics.mean_absolute_error(y_test, y_pred))
#MSE
print(metrics.mean_squared_error(y_test, y_pred))
#RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



#Nun da das Modell fertig trainiert ist, soll es auf neue Daten angewendet werden die es vorher noch nicht gesehen hat

test_data = pd.read_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Testdaten/TestdatenfürfertigesModellStündlicheDaten.csv", index_col = 0)
#Erneut alle Schritte durchführen für den Datensatz, um die Formeln des Küsteningenierswesens anzuwenden
test_data.describe()
test_data.info()

#Hinzufügen des Jahres als Spalte
test_data['Jahr'] = pd.DatetimeIndex(test_data.index).year
test_data['Monat'] = pd.DatetimeIndex(test_data.index).month
conditions1 = [
    (test_data['Monat'] == 1),
    (test_data['Monat'] == 2),
    (test_data['Monat'] == 3),
    (test_data['Monat'] == 4),
    (test_data['Monat'] == 5),
    (test_data['Monat'] == 6),
    (test_data['Monat'] == 7),
    (test_data['Monat'] == 8),
    (test_data['Monat'] == 9),
    (test_data['Monat'] == 10),
    (test_data['Monat'] == 11),
    (test_data['Monat'] == 12)
    ]

values1 = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August','September', 'Oktober', 'November','Dezember']

#Hinzufügen des Monats als Zahl und als Wort je Reihe
test_data['MonateAlsWort'] = np.select(conditions1, values1)

#Setzen der Meerestemperatur pro Monat als Durchschnittswert
valuesTemp = [3,2,3,6,11,15,18,18,16,13,9,6]

#Hinzufügen als Variable im Datensatz
test_data['SeaTemperature'] = np.select(conditions1, valuesTemp)

#Umrechnung der Km/h in m/s
test_data['GeschwInM/S'] = test_data['wspd']/3.6 

#Berechnung der Luft Wsser Temperaturdifferenz
test_data['LuftWasserDiff'] = test_data['temp']-test_data['SeaTemperature'] 

#data = pd.DataFrame(data)
#Letzte benötigte DAten angeben die konstant bleiben

WindgeschwindigkeitLand = pd.DataFrame(test_data['GeschwInM/S'])
 #m/s
#Dauer der Windstärke: 1h 20min, in Minuten: 72
Einwirkzeit = 60 #in Minuten sind hier 1,20h 72 wird dann im folgenden aug 60 gesetzt da für jede h ein Wert geschätzt wird
#Windrichtung: Südostwind
Messhöhe = 2 #m über Land
Lufttemperatur = test_data['temp'] #Grad Celcius
Wassertemperatur = test_data['SeaTemperature']
#Erdbeschleunigung
g = 9.81
#Berechnung einzelner wichtiger Parameter und Werte zur Umrechnung:
test_data['WindgeschwInMesshöhe'] = test_data['GeschwInM/S']*((10/Messhöhe)**(1/7))


#Umrechnung der Windgeschwindigkeit über Land zu über Wasser da dieser dort schneller ist
          
#Anlegen der Umrechnungsfaktoren als Variable für Berechnung der Windgeschwindigkeit über WAsser mit Temperaturdifferenz
conditions5 = [
    (test_data['LuftWasserDiff'] == 0),
    ((test_data['LuftWasserDiff'] > 0) & (test_data['LuftWasserDiff']<= 5)),
    ((test_data['LuftWasserDiff'] <= 10) & (test_data['LuftWasserDiff']> 5)),
    ((test_data['LuftWasserDiff'] <= 15) & (test_data['LuftWasserDiff']> 10)),
    ((test_data['LuftWasserDiff'] <= 20) & (test_data['LuftWasserDiff'] >15)),
    (test_data['LuftWasserDiff'] > 20),
    ((test_data['LuftWasserDiff'] <0) & (test_data['LuftWasserDiff']>= -5)),
    ((test_data['LuftWasserDiff'] < -5) & (test_data['LuftWasserDiff']>= -10)),
    ((test_data['LuftWasserDiff'] < 10) & (test_data['LuftWasserDiff']>= -15)),
    ((test_data['LuftWasserDiff'] < 15) & (test_data['LuftWasserDiff']>= -20)),
    (test_data['LuftWasserDiff'] < 20)
    ]

values5 = [1,0.92,0.85,0.8,0.78,0.77,1.05,1.15,1.18,1.21,1.23]
test_data['UmrechnungWindWassermitLuftDiff'] = np.select(conditions5,values5)

conditions3 = [
    (test_data['GeschwInM/S'] <18.5),
    (test_data['GeschwInM/S'] >= 18.5)]
values3 = [test_data['GeschwInM/S'],18.3]

test_data['InputWindUmrechnung'] = np.select(conditions3,values3) #als Input deklariert

WindgeschwindigkeitLand['FertigerFormelInput'] = test_data['InputWindUmrechnung']
WindgeschwindigkeitLand['Neu'] = 2.3-1.10488*np.log10(WindgeschwindigkeitLand['FertigerFormelInput'])

#InputwindUmrechnung muss zu float converted werden
test_data['InputWindUmrechnung'].dtypes

WindgeschwindigkeitLand.reset_index(drop =True,inplace=True)


#Zurücksetzten des Index 
#data.reset_index(drop=True, inplace=True)

#Umrechnungsfaktor der noch auf die Windgeschwindigkeit angewendet werden muss
test_data['WindüberWasserFaktor'] = 2.3-1.10488*np.log10(test_data['InputWindUmrechnung'])
#Windgeschwindigkeit über Wasser aber noch ihne Einberechnung der Luft-Wasser Temperaturdifferenz
test_data['WindgeschwüberWasser'] = test_data['WindgeschwInMesshöhe']*test_data['WindüberWasserFaktor']




#UmrechnungWind(WindgeschwindigkeitLand)
LuftWasserDifferenz = test_data['LuftWasserDiff']

#Umrechnung der WindgeschwindigkeitWasser unter Beachtung der Temperaturdifferenz


test_data['WindgeschwFertig'] = test_data['WindgeschwüberWasser']*test_data['UmrechnungWindWassermitLuftDiff']



#Berechnung Windspannungsfaktor nach Resio und Vincent

test_data['Windspannungsfaktor'] = ((0.53+0.047*test_data['WindgeschwFertig'])**0.5)*test_data['WindgeschwFertig']


#Berechnung des Fetches (Einwirkweg der Windgeschwindigkeit)
#Dimensionsloser Fetch zunächst

#Zeitumrechnung in Sekunden:
ZeitinSekunden = Einwirkzeit *60

test_data['FetchDimlos'] = (1.752*(10)**-3)*(((g*ZeitinSekunden)/test_data['Windspannungsfaktor'])**(3/2))

#Fetch in m berechnen

test_data['FetchinM'] = ((test_data['FetchDimlos']*(test_data['Windspannungsfaktor'])**2)/g)
#print("Der Fetch in m im Tiefwasser lautet:", FetchinM)

#Wellenhöhe berchnen:

test_data['Wellenhöhe'] = 0.0016*(((g*test_data['FetchinM'])/((test_data['Windspannungsfaktor'])**2))**0.5)*(((test_data['Windspannungsfaktor'])**2)/g)

#Wellenperiode berechnen:
test_data['Wellenperiode'] = 0.2857*(((g*test_data['FetchinM'])/((test_data['Windspannungsfaktor'])**2))**(1/3))*(test_data['Windspannungsfaktor']/g)

#test_data.to_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Testdaten/TestdatenfürfertigesModellStündlicheDaten_mitallenVariablen.csv")
test_data.info()

#Vergleich der errechneten Werte mit denen die dvom SVR predicted werden
test_data_fürSVR = pd.read_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Testdaten/TestdatenfürfertigesModellStündlicheDaten_mitallenVariablen.csv", index_col = 0)

#test_data_fürSVR = data.dropna(axis = 0, how = 'any')
test_data_fürSVR.info()

test_data_fürSVR.reset_index(drop=True, inplace=True)
Rechnung = test_data_fürSVR['Wellenhöhe']


test_data_fürSVR.reset_index(drop=True, inplace=True)
#Wellenhöhe_test_data_fürSVR = test_data_fürSVR['Wellenhöhe']
test_data_fürSVR.drop('Wellenhöhe', axis = 1, inplace = True)
test_data_fürSVR.drop('MonateAlsWort', axis = 1, inplace = True)
test_data_fürSVR.drop('prcp', axis = 1, inplace = True)
test_data_fürSVR.drop('coco', axis = 1, inplace = True)
test_data_fürSVR.drop('snow', axis = 1, inplace = True)
test_data_fürSVR.drop('wpgt', axis = 1, inplace = True)

test_data_fürSVR['tsun'].fillna(test_data_fürSVR['tsun'].mean(),inplace = True)

test_data_fürSVR['Wellenhöhe'] = regressor.predict(test_data_fürSVR)

Wellenhöhe_Predicted = test_data_fürSVR['Wellenhöhe']

len(Wellenhöhe_Predicted)
len(Rechnung)

frame = {'Rechnung': Rechnung, 'Wellenhöhe_Modell':Wellenhöhe_Predicted}

Vergleich = pd.DataFrame(frame)

#MAE
print(metrics.mean_absolute_error(Vergleich['Rechnung'], Vergleich['Wellenhöhe_Modell']))
#MSE
print(metrics.mean_squared_error(Vergleich['Rechnung'], Vergleich['Wellenhöhe_Modell']))
#RMSE
print(np.sqrt(metrics.mean_squared_error(Vergleich['Rechnung'], Vergleich['Wellenhöhe_Modell'])))

Vergleich.plot(y=['Rechnung'])
plt.xticks(rotation=45)
plt.show()

Vergleich.plot(y=['Wellenhöhe_Modell'])
plt.xticks(rotation=45)
plt.show()


Vergleich.plot(y=['Wellenhöhe_Modell', "Rechnung"])
plt.xticks(rotation=45)
plt.show()

#Vergleich.to_csv("/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte/Meereswellen Wetter Forecasting/Finaler Vergleich der Wellenhöhenwerte/FinalerVergleichWellenhöheAlsY.csv")
