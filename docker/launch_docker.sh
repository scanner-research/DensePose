nvidia-docker build -f docker/Dockerfile -t densepose:py35 .

nvidia-docker run -ti --ipc=host -v $(pwd):/DensePose --workdir=/DensePose -p 8888:8888 densepose:py35 /bin/bash

# docker login

# docker tag densepose:py35 docker.io/haotianz/densepose:py35

# docker push haotianz/densepose:py35
