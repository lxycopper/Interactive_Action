#export TRAIN_DIR="./reversion_benchmark/520-move_bucket"
export OUT_DIR="./experiments/reversion-pickup"
export inference_templates="reversion_bad_cases-v2"
export gpu_device="cuda:4"
# accelerate launch \
#     --config_file="./configs/single_gpu.yml" \
#     train.py \
#     --seed="2023" \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --train_data_dir=$TRAIN_DIR \
#     --placeholder_token="<R>" \
#     --initializer_token="and" \
#     --train_batch_size="2" \
#     --gradient_accumulation_steps="4" \
#     --max_train_steps="3000" \
#     --learning_rate='2.5e-04' --scale_lr \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps="0" \
#     --output_dir=$OUT_DIR \
#     --save_steps="3000" \
#     --checkpointing_steps="3000" \
#     --importance_sampling \
#     --denoise_loss_weight="1.0" \
#     --steer_loss_weight="0.01" \
#     --num_positives="4" \
#     --temperature="0.07" 

python inference.py \
--model_id $OUT_DIR \
--template_name $inference_templates \
--placeholder_string "<R>" \
--num_samples 10 \
--guidance_scale 7.5 \
--device $gpu_device \