python inference.py \
--model_id ./experiments/lift-219-distill-0.2 \
--template_name "reversion_test_v2" \
--placeholder_string "<R>" \
--num_samples 10 \
--guidance_scale 7.5 \
--device "cuda:5" \

# python inference.py \
# --model_id ./experiments/lift-219-distill-0.2 \
# --template_name "reversion_test_v2" \
# --placeholder_string "<R>" \
# --num_samples 10 \
# --guidance_scale 7.5 \
# --device "cuda:1" \


