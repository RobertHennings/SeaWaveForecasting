#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:52:45 2020

@author: Robert_Hennings
"""
import math
import datetime
from datetime import timezone
import pandas as pd
import numpy as np
#Begleitcode für das Projekt Meereswellenforecasting und das Berechnen weiterer Wellenparameter
#Ziel ist es die Parameter Wellenhöhe, Fetchlänge und Wellenperiode zu bestimmen

#Nötige Inputdaten sind:
#Windgeschwindigkeit und Richtung
#Dauer der Windgeschwindigkeit
#Temperatur der Luft
#Temperatur Wasser
#Messhöhe der Station
#Standort der Messstation (Land/Wasser)

#Beispielhaft mit der Dateneingabe einzelner Werte für einen Tag 
#Ausgangsdaten: 23.12.2017 für den Standort Laboe in Schleswig-Holstein
WindgeschwindigkeitLand = 11 #m/s
#Dauer der Windstärke: 1h 20min, in Minuten: 72
Einwirkzeit = 72 #in Minuten sind hier 1,20h 72 wird dann im folgenden aug 60 gesetzt da für jede h ein Wert geschätzt wird
#Windrichtung: Südostwind
Messhöhe = 2 #m über Land
Lufttemperatur = 8 #Grad Celcius
Wassertemperatur = 4 #Grad Celcius
#Erdbeschleunigung
g = 9.81
#Berechnung einzelner wichtiger Parameter und Werte zur Umrechnung:

WindgeschwInMesshöhe = WindgeschwindigkeitLand*((10/Messhöhe)**(1/7))

#Umrechnung der Windgeschwindigkeit über Land zu über Wasser da dieser dort schneller ist

def UmrechnungWind(WindgeschwindigkeitLand):
     if WindgeschwindigkeitLand<18.5:
       wert = WindgeschwindigkeitLand
       WindgeschwindigkeitWasser = 2.3-1.10488*np.log10(wert)
       return(WindgeschwindigkeitWasser)
     else:
        wert = 18.3
        WindgeschwindigkeitWasser = 2.3-1.10488*np.log10(wert)
        return(WindgeschwindigkeitWasser)
        
    
  
print(UmrechnungWind(WindgeschwindigkeitLand))

LuftWasserDifferenz = Lufttemperatur-Wassertemperatur
WindgeschwüberWasser = WindgeschwInMesshöhe*1.1
#Umrechnung der WindgeschwindigkeitWasser unter Beachtung der Temperaturdifferenz

WindgeschwFertig = 0.99*WindgeschwüberWasser
print("Die Windgeschwindigkeit über Wasser in m/s beträgt:",WindgeschwFertig)
#Berechnung Windspannungsfaktor nach Resio und Vincent

Windspannungsfaktor = ((0.53+0.047*WindgeschwFertig)**0.5)*WindgeschwFertig


#Berechnung des Fetches (Einwirkweg der Windgeschwindigkeit)
#Dimensionsloser Fetch zunächst

#Zeitumrechnung in Sekunden:
ZeitinSekunden = Einwirkzeit *60

FetchDimlos = (1.752*(10)**-3)*(((g*ZeitinSekunden)/Windspannungsfaktor)**(3/2))
print("Der dimensionslose Fetch lautet:", FetchDimlos)
#Fetch in m berechnen

FetchinM = ((FetchDimlos*(Windspannungsfaktor)**2)/g)
print("Der Fetch in m im Tiefwasser lautet:", FetchinM)

#Wellenhöhe berchnen:

Wellenhöhe = 0.0016*(((g*FetchinM)/((Windspannungsfaktor)**2))**0.5)*(((Windspannungsfaktor)**2)/g)
print("die Wellenhöhe beträgt in m:", Wellenhöhe)

Wellenperiode = 0.2857*(((g*FetchinM)/((Windspannungsfaktor)**2))**(1/3))*(Windspannungsfaktor/g)

print("Die Wellenperiode beträgt in Sekunden:",Wellenperiode)












