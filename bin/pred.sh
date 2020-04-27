if [ "$1" == "" ] || [ "$1" == "help" ]; then
    echo "命令格式："
    echo "\tpred.sh <image> <model>"
    exit
fi

echo "开始预测"

MODEL=model/model-20200424174211-epoch205-acc0.4595-val0.8286.hdf5
if [ "$2" != "" ]; then
  echo "使用提供的模型：$2"
  $MODEL=$2
fi

python -m main.pred --image $1 --model $MODEL