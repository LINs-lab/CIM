python ./main.py \
-d "cifar10" \
-m "resnet18_modified" \
--depth -1 \
--ipc 1 \
--epochs 1000 \
--mix_type "cutmix"

# default and useless
# --factor 2 \
# --mix_type "vanilla" \
# --crop_method "factor" \
