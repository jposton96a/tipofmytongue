REGISTRY_HOSTNAME=478499271150.dkr.ecr.us-east-1.amazonaws.com
REGISTRY=tipofmytongue_ui
TAG=$(shell git rev-parse --short HEAD)
IMAGE=${REGISTRY_HOSTNAME}/${REGISTRY}:${TAG}
API_SERVER="172.17.0.3:8000"

build:
	echo Building into image: ${IMAGE}
	docker build -t $(IMAGE) .

publish:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(REGISTRY_HOSTNAME)
	docker push $(IMAGE)

serve:
	docker run -it --rm -p 8080:80 -e API_SERVER=$(API_SERVER) $(IMAGE)