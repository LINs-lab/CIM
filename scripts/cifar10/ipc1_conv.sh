python ./main.py \
-d "cifar10" \
-m "conv3" \
--depth 2 \
--ipc 1 \
--epochs 1000 \
--mix_type "cutmix"

# default and useless
# --factor 2 \
# --mix_type "vanilla" \
# --crop_method "factor" \
