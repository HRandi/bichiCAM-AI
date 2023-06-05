# BichiCAM AI

## Description

Bichicam-AI est un système de comptage des juveniles de poissons "bichiques" via des vidéos prises principalement à la rivière du Mat de l'ile de La Reunion.

## Sauvegarder et actualiser le modèle de référence mr_sic_ind

python ./system/save_model.py --weights ./system/data/mr_sic_ind.weights --output ./system/lib/mr_sic_ind --model yolov4
--tiny

## Créer un répertoire

```bash
src
|__csv
|__videos
```

* csv : pour stocker les résultats de comptage en format .csv
* videos : mettre les vidéos à traiter dans ce dossier 

