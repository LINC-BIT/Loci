**GradMFL** comes from "GradMFL: Gradient Memory-Based Federated Learning for Hierarchical Knowledge Transferring Over Non-IID Data". It proposes a Gradient Memory-based Federated Learning (GradMFL) framework, which enables Hierarchical Knowledge Transferring over Non-IID Data. In GradMFL, a data clustering method is proposed to categorize Non-IID data to IID data according to the similarity. And then, in order to enable beneficial knowledge transferring between hierarchical clusters, it also presents a multi-stage model training mechanism using gradient memory, constraining the updating directions.



We set six continual learning methods on GradMFL. You can use the following command to run GradMFL:



~~~sh
```shell
cd baselines
cd FedKD  # method name
python mainGradMFL_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainGradMFL_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

python mainGradMFL_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

python mainGradMFL_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

python mainGradMFL_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

python mainGradMFL_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~

