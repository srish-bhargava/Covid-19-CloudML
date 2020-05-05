Instructions to deploy project 2 in kubernetes - Srishti Bhargava, Daniel Sabba


#1: Turn on IBM CLI

commands:
$ ibmcloud login
$ ibmcloud target -g <resource_group_name>
$ ibmcloud ks cluster ls
$ ibmcloud ks cluster config --cluster <cluster_name_or_ID>
$ ibmcloud ks cluster config --cluster <clusterID from ibmcloud website>
$ kubectl get nodes
$ kubectl config current-context


#2: Create Training Docker Image (utilize Dockerfile, makeModel.py and requirements.txt inside the training folder):

commands:
$ docker build -t neuralnet_sentiment .


#3: Push Image to Dockerhub (alerady done for TAs)

commands:
$ docker login --username=ds1952
$ docker images <get IMAGE ID>
$ docker tag <IMAGE ID> ds1952/neuralnet_sentiment:latest
$ docker push ds1952/neuralnet_sentiment:latest
$ docker tag <IMAGE ID> srishti95/svmclassifier:1.0.2
$ docker push srishti95/svmclassifier:1.0.2


#4: Train model with "kubectl create" command in IBMCloud with dockerhub image - utilize deployment-training.yaml inside the training folder)

commands:
$ kubectl create -f deployment-training_snn.yaml
$ kubectl create -f deployment-training_svm.yaml

#5: Once training is completed, create Inference Docker Image (utilize Dockerfile, inference-script-snn-svm.py and requirements.txt inside the inference folder). Once completed push imaged to dockerhub.

commands:
$ docker build -t inference_image_proj2 .
$ docker images <get IMAGE ID>
$ docker tag inference_image_proj2 ds1952/proj2:latest
$ docker push ds1952/proj2:latest


#6: Perform inference with "kubectl command" with dockerhub image

commands:
$ kubectl create -f deployment-inference.yaml


#7: Expose deployment for inference and set up a service for external access - using file hw4-service.yaml. Additionally, get IP for containers (save that IP address) and nodeport for container.

commands:
$ kubectl expose deployment deployment-inference --type=NodePort --port=8001
$ kubectl create -f service.yaml
$ kubectl get nodes -o jsonpath='{.items[*].status.addresses[?(@.type=="ExternalIP")].address}'
<COPY EXTERNAL IP>
$ kubectl get service
$ kubectl describe service/inference-service3
<COPY NODEPORT>


#8: Access training throught a webbrowser

Go to 
http://<EXTERNAL IP>:<NODEPORT>/?name="TWITTER TO TEST"
http://184.172.233.161:31472/?model_number=0&name="terrible horrific awful"

#9: KubeConfig and certificate file (.pem) found in ~/.bluemix/plugins/container-service/clusters/ have been provided in /Submission/KubeConfig_Certificates - they are called ca-hou02-mycluster-hw4.pem and kube-config-hou02-mycluster-hw4.yml - Please disregard the "hw4" reference - this comes from the fact that I created and named this cluster when doing hw4 and used it for subsequent assigments.
