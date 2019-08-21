# 参数说明：
# python -m main.train \
#    --name=attention_ocr \
#    --epochs=200 \                 # 200个epochs，但是不一定能跑完，因为由ealy stop
#    --steps_per_epoch=1000 \       # 每个epoch对应的批次数，其实应该是总样本数/批次数，但是我们的样本上百万，太慢，所以，我们只去1000个批次
#                                   # 作为一个epoch，为何要这样呢？因为只有每个epoch结束，keras才回调，包括validate、ealystop等
#    --batch=64 \
#    --learning_rate=0.001 \
#    --validation_batch=64 \
#    --validation_steps=10 \        # 这个是说你测试几个批次，steps这个词不好听，应该是batchs，实际上可以算出来，共测试64x10=640个样本
#    --workers=10 \
#    --preprocess_num=100 \
#    --early_stop=10 \              # 如果10个epochs都没提高，就停了吧，大概是1万个batch

echo "开始训练"

if [ "$1" == "console" ] || [ "$1" == "debug" ]; then
    echo "调试模式"
    python -m main.train \
    --name=attention_ocr \
    --epochs=10 \
    --steps_per_epoch=10 \
    --batch=8 \
    --learning_rate=0.001 \
    --validation_steps=2  \
    --validation_batch=8 \
    --workers=2 \
    --preprocess_num=5 \
    --early_stop=10
    exit
fi

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep attention_ocr|awk '{print $2}'|xargs kill -9
    exit
fi


Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0

echo "生产模式"
echo "使用 #$CUDA_VISIBLE_DEVICES GPU"

nohup python -m main.train \
    --name=attention_ocr \
    --epochs=1000 \
    --steps_per_epoch=1000 \
    --batch=64 \
    --learning_rate=0.001 \
    --validation_batch=64 \
    --validation_steps=10 \
    --workers=10 \
    --preprocess_num=100 \
    --early_stop=200 \
    >> ./logs/Attention_GPU$CUDA_VISIBLE_DEVICES_$Date.log 2>&1 &

