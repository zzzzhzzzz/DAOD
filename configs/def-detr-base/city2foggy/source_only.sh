BATCH_SIZE=8
DATA_ROOT=/data/dataset/zhanghaozhuo
OUTPUT_DIR=./outputs/yolo/city2foggy/source_only

python /data/ckpt/zhanghaozhuo/Doman_Adaptive_Detection/main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 50 \
--epoch_lr_drop 40 \
--mode single_domain \
--output_dir ${OUTPUT_DIR} \
--device cuda:2
