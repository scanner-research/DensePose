nvidia-docker build -f docker/Dockerfile -t densepose:py35 .

nvidia-docker run -ti --ipc=host -v $(pwd):/DensePose-py2 --workdir=/DensePose-py2 -p 8888:8888 densepose:py35 /bin/bash

#docker login

#docker tag dip_video_inpainting:CUDA9-py35-pytorch0.4.1 docker.io/haotianz/dip_video_inpainting:CUDA9-py35-pytorch0.4.1

#docker push haotianz/dip_video_inpainting:CUDA9-py35-pytorch0.4.1
