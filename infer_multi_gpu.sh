python2 tools/test_net.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    NUM_GPUS 2