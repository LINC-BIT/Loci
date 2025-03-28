"""
Re-implementation of PackNet Continual Learning Method
"""
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from copy import deepcopy
def MultiClassCrossEntropy(logits, labels, t,T=2):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2




class UserpFedGen():
    def __init__(self,model, generative_model,available_labels,args=None):
        # super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.model = deepcopy(model)
        self.local_epochs = args.local_ep
        self.lr = args.lr
        self.beta = 1

        # self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)


        self.unique_labels = 10
        self.generative_alpha = 10
        self.generative_beta = 0.1

        # those parameters are for personalized federated learning.
        self.local_model = deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()

        # self.optimizer = torch.optim.Adam(
        #     params=self.model.parameters(),
        #     lr=self.learning_rate, betas=(0.9, 0.999),
        #     eps=1e-08, weight_decay=1e-2, amsgrad=False)


        self.label_counts = {}
        self.gen_batch_size = 32
        self.generative_model = generative_model
        self.latent_layer_idx = -1
        self.available_labels = available_labels
        self.device = args.device
        self.batch_size = args.local_bs

    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer
    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels):
        for label in labels:
            self.label_counts[int(label)] += 1

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self,t, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        self.generative_model.eval()
        self.generative_model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for epoch in range(self.local_epochs):
            for images,targets in self.tr_dataloader:
                self.update_label_counts(targets - 10 *t)
                self.optimizer.zero_grad()
                #### sample from real dataset (un-weighted)
                images = images.to(self.device)
                targets = (targets - 10 * t).cuda()
                offset1, offset2 = compute_offsets(t, 10)

                user_output_logp=self.model.forward(images, t)[:, offset1:offset2]
                predictive_loss=self.ce_loss(user_output_logp, targets)

                #### sample y and generate z
                if regularization and epoch < early_stop:
                    generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    ### get generator output(latent representation) of the same label
                    gen_output=self.generative_model(targets, latent_layer_idx=self.latent_layer_idx,device=self.device)['output']
                    logit_given_gen=self.model(gen_output, t,start_layer_idx=self.latent_layer_idx)[:, offset1:offset2]
                    target_p=F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss= generative_beta * self.ensemble_loss(F.log_softmax(user_output_logp,dim=1), target_p)

                    sampled_y=np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y=torch.tensor(sampled_y).to(self.device)
                    gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx,device=self.device)
                    gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                    user_output_logp =self.model(gen_output, t,start_layer_idx=self.latent_layer_idx)[:, offset1:offset2]
                    teacher_loss =  generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(F.log_softmax(user_output_logp,dim=1), sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss=predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS+=teacher_loss
                    LATENT_LOSS+=user_latent_loss
                else:
                    #### get loss and perform optimization
                    loss=predictive_loss
                loss.backward()
                self.optimizer.step()#self.local_model)
            loss, acc = self.eval(t)
            print('acc:',acc)
        # local-model <=== self.model
        # self.clone_model_paramenter(self.model.parameters(), self.local_model)
        # if personalized:
        #     self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        if regularization:
            TEACHER_LOSS=TEACHER_LOSS.cpu().detach().numpy() / (self.local_epochs)
            LATENT_LOSS=LATENT_LOSS.cpu().detach().numpy() / (self.local_epochs)
            info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
            print(info)
        loss,acc = self.eval(t)
        return loss,acc

    def eval(self, t, train=True, model=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        if train:
            dataloaders = self.tr_dataloader
        if model is None:
            model = self.model
        # Loop batches
        model.eval()
        with torch.no_grad():
            for images, targets in dataloaders:
                images = images.cuda()
                targets = (targets - 10 * t).cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = model.forward(images, t)[:, offset1:offset2]

                loss = self.ce_loss(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task,r,early_stop=10,regularization =r>0)
    print('-' * 100)
    from random import sample
    samplenum = 4
    if samplenum < t:
        samplenum = t


    return appr.model.state_dict(), loss,appr.label_counts, 0