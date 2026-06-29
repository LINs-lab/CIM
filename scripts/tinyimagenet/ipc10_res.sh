python ./main.py \
-d "tinyimagenet" \
-m "resnet18_modified" \
--depth -1 \
--ipc 10 \
--epochs 1000 \
--mix_type "cutmix"

# default and useless
# --factor 2 \
# --mix_type "vanilla" \
# --crop_method "factor" \
