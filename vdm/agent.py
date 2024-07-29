import pdb

import numpy as np
import torch
from easydl import *
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
import copy

import clip
from net import *
from utils import get_optimizer
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


_MODELS = {
    'RN50': 'RN50',
    'RN101': 'RN101',
    'ViTB16': 'ViT-B/16',
    'ViTB32': 'ViT-B/32'
}


class ClfAgent:
    def __init__(self, opt, train_loader, test_loader, nclass, optim_conf, device, proto=None, scheduler=None, input_img=True):
        self.epochs = opt.classifier.epochs
        self.input_dim = opt.dataset.image_embedding_dim
        self.metric = opt.classifier.metric
        self.temperature = opt.classifier.temperature
        self.weight_activation = opt.classifier.weight_activation
        self.train_loader = train_loader # dataloader of tensor
        self.test_loader = test_loader # dataloader of tensor
        self.nclass = nclass
        self.device = device
        self.proto = proto
        self.input_img = input_img
        if self.input_img:
            clip_model, _, _ = clip.load(_MODELS[opt.dataset.image_embedding], self.device, jit=False)
            self.image_encoder = clip_model.visual.float()
            del clip_model.transformer
            torch.cuda.empty_cache()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self.clf = Classifier(self.input_dim, nclass, self.metric, self.proto, self.weight_activation).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = get_optimizer(self.clf.parameters(), optim_conf)
        self.scheduler = scheduler
        if scheduler is not None:
            self.optimizer = OptimWithSheduler(optimizer=self.optimizer, scheduler_func=scheduler)
        self.steps = 0

    def train(self, epoch):
        self.clf.train()
        iters = tqdm(self.train_loader, desc=f'epoch {epoch} ', total=len(self.train_loader))

        for i, (image, label) in enumerate(iters):
            self.steps += 1
            image = image.to(self.device)
            if self.input_img:
                feat = self.image_encoder(image)
            else:
                feat = image
            label = label.to(self.device)

            tem = self.temperature / (1 + np.log(self.steps))
            logits = self.clf(feat, tem)
            loss = self.criterion(logits, label)

            with OptimizerManager([self.optimizer]):
                loss.backward()

    def validate(self, test_loader):
        self.clf.eval()
        iters = tqdm(test_loader, desc=f'testing ', total=len(test_loader))
        start = 0
        prediction = torch.LongTensor(len(test_loader.dataset))
        ground_truth = torch.LongTensor(len(test_loader.dataset))
        with torch.no_grad():
            for i, (image, label) in enumerate(iters):
                image = image.to(self.device)
                if self.input_img:
                    feat = self.image_encoder(image)
                else:
                    feat = image
                label = label.to(self.device)
                logits = self.clf(feat)
                # measure accuracy and record loss
                end = start + feat.size(0)
                _, prediction[start:end] = torch.max(logits.data, 1)
                ground_truth[start:end] = label.data
                start = end
        acc = self.compute_per_class_acc(ground_truth, prediction)
        return acc

    def fit(self):
        best_acc = 0.
        acc = self.validate(self.test_loader)
        print(f'Init top-1 accuracy = {acc}')
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            acc = self.validate(self.test_loader)
            print(f'Epoch [{epoch}/{self.epochs}]: top-1 accuracy = {acc}')
            if acc >= best_acc:
                best_acc = acc
                clf_dict = copy.deepcopy(self.clf.state_dict())

        self.best_acc = best_acc
        self.clf_dict = clf_dict
        print(f'Final : top-1 best accuracy = {best_acc}')

    def compute_per_class_acc(self, ground_truth, prediction):
        classes = ground_truth.unique()
        acc_per_class = torch.FloatTensor(classes.size(0)).fill_(0)
        for i in range(classes.size(0)):
            idx = (ground_truth == classes[i])
            acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
        return acc_per_class.mean()


class SimAgent:
    def __init__(self, epochs, feature_activation, teacher, nclass, ndim, nsample, optim_conf, device,
                 temperature=1.0, scheduler=None, proto=None, concentration=None, regularizer=None, blackbox=True):
        self.epochs = epochs
        self.nclass = nclass
        self.ndim = ndim
        self.nsample = nsample
        self.activation = utils.activation_dict[feature_activation]
        self.device = device
        self.temperature = temperature

        self.regularizer = regularizer
        if regularizer is None:
            self.simulator = Simulator(nclass, ndim, proto=proto, concentration=concentration, learned_concentration=False).to(self.device)
        else:
            self.simulator = Simulator(nclass, ndim, proto=proto, concentration=concentration, learned_concentration=True).to(self.device)
        self.sim_params = self.simulator.state_dict()
        self.blackbox = blackbox

        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = get_optimizer(self.simulator.parameters(), optim_conf)
        self.scheduler = scheduler
        if scheduler is not None:
            self.optimizer = OptimWithSheduler(optimizer=self.optimizer, scheduler_func=scheduler)

    def fit(self):
        self.simulator.train()
        global_epochs = tqdm(range(self.epochs), desc='simulator train')
        best_protocol = 0.
        for epoch in range(1, self.epochs + 1):
            features = []
            labels = []
            mu, concentration = self.simulator.forward()

            if self.activation is not None:
                mu = self.activation(mu)

            distribution = VonMisesFisher(mu, concentration)

            for i in range(self.nsample):
                feature = distribution.rsample()
                features.append(feature)
                label = torch.arange(self.nclass).to(self.device)
                labels.append(label)

            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            if self.activation is not None:
                features = self.activation(features)

            if not self.blackbox:
                teacher_logits = self.teacher(features, self.temperature)
                loss = F.cross_entropy(teacher_logits, labels)
            else:
                teacher_logits = (self.teacher(features, self.temperature)).detach()
                student_logits = F.normalize(features) @ F.normalize(mu).t() / self.temperature
                loss = F.mse_loss(student_logits, teacher_logits)
                # loss = F.kl_div(student_logits.log_softmax(1), teacher_logits.softmax(1))

            protocol = self.teacher(mu).diag().mean()
            if protocol > best_protocol:
                best_protocol = protocol
                self.sim_params = self.simulator.state_dict()

            with OptimizerManager([self.optimizer]):
                loss.backward()

            global_epochs.update()

        self.best_protocol = best_protocol
        print(self.best_protocol)


class GenAgent:
    def __init__(self, epochs, feature_activation, teacher, condition_dim, noise_dim, hidden_dim, output_dim, nsample,
                 optim_conf, device, conditions, metric, gen_epochs=1, disdill_epochs=1, temperature=1.0, blackbox=True):
        self.conditions = conditions # FloatTensor: size([nclass, image_embedding_dim])
        self.nclass = self.conditions.size(0)
        self.epochs = epochs
        self.gen_epochs = gen_epochs
        self.disdill_epochs = disdill_epochs
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.feature_activation = feature_activation
        self.device = device

        self.generator = Generator(condition_dim, noise_dim, hidden_dim, output_dim).to(self.device)
        self.activation = utils.activation_dict[feature_activation]

        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.metric = metric
        self.student = Classifier(self.output_dim, self.nclass, self.metric, self.conditions, self.feature_activation).to(self.device)

        self.nsample = nsample
        self.blackbox = blackbox
        self.temperature = temperature
        self.optimizerG = get_optimizer(self.generator.parameters(), optim_conf)
        self.optimizerS = get_optimizer(self.student.parameters(), optim_conf)

    def fit(self):
        global_epochs = tqdm(range(self.epochs), desc='generator train')
        label_pool = np.array([l for l in range(self.conditions.size(0))])
        for epoch in range(1, self.epochs + 1):

            self.generator.train()
            for i in range(self.gen_epochs):
                batch_size = self.nsample * self.nclass
                label = np.random.choice(a=label_pool, size=batch_size, replace=True)
                condition = self.conditions[label]
                noise = torch.randn(batch_size, self.noise_dim)

                condition = condition.to(self.device)
                noise = noise.to(self.device)
                labels = torch.from_numpy(label).long().to(self.device)

                features = self.generator(condition, noise)
                if self.activation is not None:
                    features = self.activation(features)

                if not self.blackbox:
                    teacher_logits = self.teacher(features, self.temperature)
                    loss = F.cross_entropy(teacher_logits, labels)
                else:
                    teacher_logits = self.teacher(features, self.temperature).detach()
                    student_logits = self.student(features, self.temperature)
                    loss = F.mse_loss(student_logits, teacher_logits)

                with OptimizerManager([self.optimizerG, self.optimizerS]):
                    loss.backward()

            self.generator.eval()
            for j in range(self.disdill_epochs):
                batch_size = self.nsample * self.nclass
                label = np.random.choice(a=label_pool, size=batch_size, replace=True)
                condition = self.conditions[label]
                noise = torch.randn(batch_size, self.noise_dim)

                condition = condition.to(self.device)
                noise = noise.to(self.device)
                labels = torch.from_numpy(label).long().to(self.device)

                features = self.generator(condition, noise).detach()
                if self.activation is not None:
                    features = self.activation(features)

                teacher_logits = self.teacher(features, self.temperature).detach()
                preds = teacher_logits.topk(1, 1, True, True)[1].t().squeeze()
                idx = (preds == labels)
                student_logits = self.student(features[idx], self.temperature)
                loss = F.mse_loss(student_logits, teacher_logits[idx])

                with OptimizerManager([self.optimizerS]):
                    loss.backward()

            global_epochs.update()
