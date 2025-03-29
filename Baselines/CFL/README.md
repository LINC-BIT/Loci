**CFL** comes from "Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints". It presents clustered FL, a novel federated multitask learning (FMTL) framework, which exploits geometric properties of the FL loss surface to group the client population into clusters with jointly trainable data distributions. In contrast to existing FMTL approaches, CFL does not require any modifications to the FL communication protocol to be made, is applicable to general nonconvex objectives (in particular, deep neural networks), does not require the number of clusters to be known a priori, and comes with strong mathematical guarantees on the clustering quality. CFL is flexible enough to handle client populations that vary over time and can be implemented in a privacy-preserving way. As clustering is only performed after FL has converged to a stationary point, CFL can be viewed as a postprocessing method that will always achieve greater or equal performance than conventional FL by allowing clients to arrive at more specialized models.



We set six continual learning methods on CFL. You can use the following command to run CFL:



~~~sh
```shell
cd baselines
cd CFL  # method name
python mainCFL_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainCFL_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

python mainCFL_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

python mainCFL_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

python mainCFL_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

python mainCFL_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~

