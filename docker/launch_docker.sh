# nvidia-docker build -f docker/Dockerfile -t densepose:py2 .

nvidia-docker run -ti --ipc=host -v $(pwd):/Projects --workdir=/Projects -p 8888:8888 densepose:py2 /bin/bash

#docker login

#docker tag dip_video_inpainting:CUDA9-py35-pytorch0.4.1 docker.io/haotianz/dip_video_inpainting:CUDA9-py35-pytorch0.4.1

#docker push haotianz/dip_video_inpainting:CUDA9-py35-pytorch0.4.1
