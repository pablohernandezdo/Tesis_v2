#!/bin/bash

# Correr prueba.py con los parametros necesarios
# Aqui modificar el nombre del modelo que se haya entrenado
echo "Training CBN model on $trn and $tst datasets"
python ../prueba.py --classifier LSTM --model_name LSTM_1epch_1e4_16
