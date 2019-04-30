python2 tools/infer_video.py \
    --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
    --data-dir /Projects/esper_haotian/esper/app/data \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
    --input-video Tabletennis_2012_Olympics_men_single_final_gold.mp4
