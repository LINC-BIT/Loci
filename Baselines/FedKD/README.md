**FedKD** comes from "Communication-efficient federated learning via knowledge distillation". we present a federated learning method named FedKD that is both communicationefficient and effective, based on adaptive mutual knowledge distillation and dynamic gradient compression techniques. It is validated on three different scenarios that need privacy protection, showing that it maximally can reduce 94.89% of communication cost and achieve competitive results with centralized model learning. It provides a potential to efficiently deploy privacy-preserving intelligent systems in many scenarios, such as intelligent healthcare and personalization.



We set six continual learning methods on FedMD. You can use the following command to run FedKD:



~~~sh
```shell
 	cd baselines
  cd FedKD  # method name
	python mainKD_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

​	python mainKD_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

​	python mainKD_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

​	python mainKD_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

​	python mainKD_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

​	python mainKD_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~

