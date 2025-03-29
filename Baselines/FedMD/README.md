**FedMD** comes from "Communication-efficient learning of deep networks from decentralized data". It presents a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10–100× as compared to synchronized stochastic gradient descent.



We set six continual learning methods on FedMD. You can use the following command to run FedMD:



	```shell

 	cd baselines
  	cd FedMD  # method name
	python mainMD_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

​	python mainMD_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

​	python mainMD_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

​	python mainMD_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

​	python mainMD_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

​	python mainMD_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate

​	```

