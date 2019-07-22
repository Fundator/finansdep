# Verdifastsettelse av fritidsboliger

## Instruksjoner for å kunne bruke notebook
- [Last ned og installer anaconda for python 3.7 for ditt operativsystem](https://www.anaconda.com/distribution/). Følg anbefalte innstillinger
- Last ned eller klon dette repoet

Når man har installert anaconda, åpne programmet **anaconda prompt** og skriv inn følgende kommando

~~~~
conda env create -f environment.yml
~~~~

Man kan så liste opp alle conda enviromnents ved hjelp av følgende kommando

~~~~
conda env list
~~~~

Man vil nå en liste med opptil flere enviroments, og for å velge enviroment for å kunne kjøre denne notebooken, bytter vi til det nye environmentet vi nettop lagde ut ifra .yml filen ved å skrive følgende kommando

~~~~
conda activate finansdep
~~~~

Man kan så starte jupyter notebook ved å skrive inn følgende kommando i anaconda prompt:

~~~~
jupyter notebook
~~~~

Man vil da åpne en ny fane i nettleseren som kjører jupyter notebook. Hvis denne fanen ikke åpnes, kan du skrive følgende adresse i adressefeltet til nettleseren:  `http://localhost:8888/` 
Naviger deg til mappen der filen med navn 'Finansdepartementet.ipynb' ligger, og åpne denne. Man kan så eksekvere celler i jupyter notebook sekvensielt


