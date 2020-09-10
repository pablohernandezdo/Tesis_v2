#!/bin/bash

# Correr prueba.py con los parametros necesarios
# Aqui modificar el nombre del modelo que se haya entrenado
echo "Training ANN model on $trn and $tst datasets"
python ../prueba.py --classifier C --model_name ANN_10epch
