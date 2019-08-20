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
    --epochs=200 \
    --steps_per_epoch=1000 \
    --batch=64 \
    --learning_rate=0.001 \
    --validation_batch=64 \
    --validation_steps=10 \
    --workers=10 \
    --early_stop=10 \
    >> ./logs/Attention_GPU$CUDA_VISIBLE_DEVICES_$Date.log 2>&1 &
