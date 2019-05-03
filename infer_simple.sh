python3 tools/infer_simple.py \
    --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/demo_im.jpg