model_path=$1

python3 torch2tf.py --input "$model_path"--output age-model --model-class model:get_model --input-shape 1,3,224,224

