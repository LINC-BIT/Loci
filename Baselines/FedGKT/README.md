**FedGKT** comes from "Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge". It designs a variant of the alternating minimization approach to train small CNNs on edge nodes and periodically transfer their knowledge by knowledge distillation to a large server-side CNN. FedGKT consolidates several advantages into a single framework: reduced demand for edge computation, lower communication bandwidth for large CNNs, and asynchronous training, all while maintaining model accuracy comparable to FedAvg. We train CNNs designed based on ResNet-56 and ResNet-110 using three distinct datasets (CIFAR-10, CIFAR-100, and CINIC-10) and their non-I.I.D. variants. 



We set six continual learning methods on FedGKT. You can use the following command to run FedGKT:



~~~sh
```shell
cd baselines
cd FedGKT  # method name
python mainGKT_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainGKT_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

python mainGKT_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

python mainGKT_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

python mainGKT_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

python mainGKT_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~

