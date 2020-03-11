model_path=$1
output_path=$2

if [ -z "$output_path" ];
then
  output_path=age-model
fi

python3 torch2tf.py --input "$model_path" --output "$output_path" --model-class model:get_model --input-shape 1,3,224,224

