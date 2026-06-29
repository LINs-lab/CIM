python ./main.py \
-d "tinyimagenet" \
-m "conv4" \
--depth 3 \
--ipc 50 \
--epochs 1000 \
--mix_type "cutmix"

# default and useless
# --factor 2 \
# --mix_type "vanilla" \
# --crop_method "factor" \
