import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datasets.image_util as util
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import sys
import copy
import pdb
import math


class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netDec=None, dec_size=4096, dec_hidden_size=4096, ratio=1.0):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone()
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        # self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.model = COS_LOGSOFTMAX_CLASSIFIER(data_loader.attribute)
        # self.netDec = netDec
        self.netDec = None
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        log_p_y = torch.zeros(data_loader.nclasses)  # class prior
        p_y_seen = torch.zeros(data_loader.nseenclasses)  # conditional class prior on seen class (near Eq. 14)
        p_y_unseen = torch.ones(data_loader.nunseenclasses) / data_loader.nunseenclasses  # conditional class prior on unseen class (near Eq. 14)
        for i in range(p_y_seen.size(0)):
            iclass = data_loader.seenclasses[i]
            index = (self.train_Y == iclass)
            p_y_seen[i] = index.sum().float()
        p_y_seen /= p_y_seen.sum()
        log_p_y[data_loader.seenclasses] = p_y_seen.log()
        log_p_y[data_loader.unseenclasses] = p_y_unseen.log()
                
        log_p0_Y = torch.zeros(data_loader.nclasses)  # seen-unsee prior (Eq. 11)
        p0_s = 1 / (1 + 1 / ratio)
        p0_u = (1 - p0_s)
        log_p0_Y[data_loader.seenclasses] = math.log(p0_s)
        log_p0_Y[data_loader.unseenclasses] = math.log(p0_u)
        self.log_p_y = log_p_y
        self.log_p0_Y = log_p0_Y

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.log_p_y = self.log_p_y.cuda()
            self.log_p0_Y = self.log_p0_Y.cuda()

        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.acc_base, self.acc_novel, self.HM, self.epoch= self.fit()
        else:
            self.acc_per_class, self.acc, self.best_model = self.fit_zsl()


    def fit_zsl(self):
        best_acc_per_class = 0
        best_acc = 0
        mean_loss = 0
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                output = self.model(self.input)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
            acc_per_class, acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc_per_class > best_acc_per_class:
                best_acc_per_class = acc_per_class
                best_model = copy.deepcopy(self.model.state_dict())

            if acc > best_acc:
                best_acc = acc

        return best_acc_per_class, best_acc, best_model
        
    def fit(self):
        best_seen, best_unseen, best_H = 0., 0., 0.
        best_base, best_novel, best_HM = 0., 0., 0.
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                logits = self.model(self.input)
                # logits = logits + self.log_adjusted_weight[batch_label, :]
                logits = logits + self.log_p_y + self.log_p0_Y
                # loss = self.criterion(output, self.label)
                loss = F.cross_entropy(logits, self.label)
                loss.backward()
                self.optimizer.step()
            acc_seen, _ = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, _ = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            _, acc_base = self.val_base2new(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            _, acc_novel = self.val_base2new(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            HM = 2*acc_base*acc_novel / (acc_base+acc_novel)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
            if HM > best_HM:
                best_base = acc_base
                best_novel = acc_novel
                best_HM = HM

        return best_seen, best_unseen, best_H, best_base, best_novel, best_HM, epoch


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    @torch.no_grad()
    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = test_X[start:end].cuda()
            else:
                inputX = test_X[start:end]
            output = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc_per_class = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        acc = torch.sum(predicted_label == test_label) / test_label.size(0)
        return acc_per_class, acc



    @torch.no_grad()
    def val_base2new(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        test_label = util.map_label(test_label, target_classes)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = test_X[start:end].cuda()
            else:
                inputX = test_X[start:end]
            output = self.model(inputX)
            output = output[:, target_classes]
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc_per_class = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        acc = torch.sum(predicted_label == test_label) / test_label.size(0)
        return acc_per_class, acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        target_classes = np.unique(test_label.numpy())
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= len(target_classes)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = test_X[start:end].cuda()
            else:
                inputX = test_X[start:end]
            output = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc_per_class = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        acc = torch.sum(predicted_label == util.map_label(test_label, target_classes)) / test_label.size(0)
        return acc_per_class, acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() 


    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = test_X[start:end].cuda()
            else:
                inputX = test_X[start:end]
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o


class COS_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, proto, tem=0.01):
        super(COS_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Parameter(proto)
        self.tem = tem
    def forward(self, x):
        x = F.normalize(x)
        proto = F.normalize(self.fc)
        o = x @ proto.t() / self.tem
        return o
