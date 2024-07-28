import math
import sys,os,pdb
d = os.path.dirname(__file__)
parent_path = os.path.dirname(os.path.abspath(d))
sys.path.append(parent_path)
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from easydl import *
import random
import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import iv
from scipy.stats import vonmises
from scipy.optimize import root_scalar
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from yaml_config import *
from dataset import *
from lr_schedule import *
import agent
import net
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import compute_covariance, activation_dict, get_multivariate_normal_samplers
from preprocess import *


def train_classifier(x, y, proto=None):
    print('training the classifier')
    # prepare datasets and dataloaders
    mapped_y = map_label(y, torch.from_numpy(seenclasses).long())
    virtual_train_seen_set = SimpleFeatureSet(x, mapped_y)
    virtual_train_label_list = mapped_y.tolist()
    virtual_train_freq = Counter(virtual_train_label_list)
    virtual_train_sample_weight = {i: 1.0 / virtual_train_freq[i] if opt.dataset.resample else 1.0 for i in virtual_train_freq}
    virtual_train_sample_weights = [virtual_train_sample_weight[i] for i in virtual_train_label_list]
    sampler = torch.utils.data.WeightedRandomSampler(virtual_train_sample_weights, len(virtual_train_label_list))
    virtual_train_seen_loader = torch.utils.data.DataLoader(dataset=virtual_train_seen_set,
                                                            batch_size=opt.train.batch_size, sampler=sampler,
                                                            num_workers=opt.dataset.workers,
                                                            drop_last=True)

    mapped_test_seen_label = map_label(torch.from_numpy(test_seen_label).long(), torch.from_numpy(seenclasses).long())
    test_seen_set = SimpleFeatureSet(torch.from_numpy(test_seen_feature).float(), mapped_test_seen_label)
    test_seen_loader = torch.utils.data.DataLoader(dataset=test_seen_set,
                                              batch_size=opt.test.batch_size, shuffle=False,
                                              num_workers=opt.dataset.workers,
                                              drop_last=False)

    if opt.classifier.optimizer == 'SGD':
        clf_optim_conf = [
            'SGD',
            opt.classifier.optimconf.lr,
            opt.classifier.optimconf.momentum,
            opt.classifier.optimconf.weight_decay,
            opt.classifier.optimconf.dampening,
            opt.classifier.optimconf.nesterov
        ]
    elif opt.classifier.optimizer == 'Adam':
        clf_optim_conf = [
            'Adam',
            opt.classifier.optimconf.lr,
            opt.classifier.optimconf.betas,
            opt.classifier.optimconf.weight_decay,
            opt.classifier.optimconf.eps,
            opt.classifier.optimconf.amsgrad
        ]
    elif opt.classifier.optimizer == 'AdamW':
        clf_optim_conf = [
            'AdamW',
            opt.classifier.optimconf.lr,
            opt.classifier.optimconf.betas,
            opt.classifier.optimconf.weight_decay,
            opt.classifier.optimconf.eps,
            opt.classifier.optimconf.amsgrad
        ]

    if opt.classifier.scheduler != 'None':
        if opt.classifier.scheduler == 'stepwise':
            clf_scheduler = lambda step, initial_lr: stepwiseDecaySheduler(step, initial_lr, gamma=0.001, power=0.75)
        elif opt.classifier.scheduler == 'inverse':
            clf_scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75,
                                                                      max_iter=10000)
    else:
        clf_scheduler = None

    seen_clf = agent.ClfAgent(opt, virtual_train_seen_loader, test_seen_loader, nseenclasses, clf_optim_conf, device, proto=proto, scheduler=clf_scheduler, input_img=False)

    seen_clf.fit()
    return seen_clf


def get_seen_prototype(typ, activation=None):
    print('get the prototype')
    if typ == 'mean':
        prototype = torch.FloatTensor(nseenclasses, opt.dataset.image_embedding_dim)
        for c in range(nseenclasses):
            cid = seenclasses[c]
            this_train_seen_feature = train_seen_feature[train_seen_label == cid]
            this_train_seen_feature = this_train_seen_feature.mean(dim=0)
            prototype[c] = this_train_seen_feature
    elif typ == 'weight':
        with open(os.path.join(log_dir, f'best_real_seen_classifier-{exp_name}.pkl'), 'rb') as f:
            seen_clf_dict = torch.load(f)
        seen_clf = net.Classifier(opt.dataset.image_embedding_dim, nseenclasses, opt.classifier.metric,
                                  opt.classifier.weight_activation)
        seen_clf.load_state_dict(seen_clf_dict['classifier'])
        weights = seen_clf.weights.cpu().data
        prototype = weights
    else: # standard
        assert False, 'Not implemented!'

    if activation is not None:
        prototype = activation(prototype)

    prototype = F.normalize(prototype)
    prototype = prototype.cpu()

    return prototype


def get_seen_concentration(prototype):
    print('get the cooncentrace')
    cos = prototype @ prototype.t()
    arc = torch.acos(cos).numpy()
    # proto = prototype.numpy()
    # distance = distance_matrix(proto, proto, 2)

    # feature = torch.from_numpy(train_seen_feature).float()
    # label = torch.from_numpy(train_seen_label).long()
    # prior_concentration = []
    # for c in seenclasses:
    #     idx = (label == c)
    #     this_feature = feature[idx] # [num, 512]
    #     this_feature_mean = this_feature.mean(dim=0, keepdim=True) # [1, 512]
    #     this_cos = F.normalize(this_feature) @ F.normalize(this_feature_mean).t()
    #     this_arc = torch.acos(this_cos).squeeze() # [num, 1]
    #     this_std = this_arc.numpy().max() / 4
    #     this_concentration = 1 / this_std ** 2
    #     prior_concentration.append(this_concentration)
    # print(prior_concentration)
    # print(max(prior_concentration))
    # pdb.set_trace()

    minc = []
    for c in range(nseenclasses):
        arc_c = arc[c, :]
        noself_arc_c = np.delete(arc_c, c)
        min_arc_c = np.min(noself_arc_c)
        max_std_c = min_arc_c / 6
        minc.append(1 / max_std_c ** 2)
        # distance_c = distance[c, :]
        # noself_distance_c = np.delete(distance_c, c)
        # min_distance_c = np.min(noself_distance_c)
        # max_std_c = min_distance_c / (2 * np.sqrt(opt.dataset.image_embedding_dim))
        # minc.append(1 / max_std_c ** 2)

    concentration = minc

    return torch.from_numpy(np.array(concentration)).float()


def sample_virtual_data(mean, concentration, num, activation=None):
    print('sample virtual data via gmm')
    concentration = concentration
    features = []
    labels = []

    distribution = VonMisesFisher(mean, concentration)

    for i in range(num):
        feature = distribution.rsample()
        feature = feature
        label = torch.from_numpy(seenclasses).long()

        features.append(feature)
        labels.append(label)

    x = torch.cat(features, dim=0).cpu()
    if activation is not None:
        x = activation(x)
    y = torch.cat(labels, dim=0).cpu()

    return x, y


def train_simulator(proto, concentration):
    print('training the simulator')
    if opt.vdm.simulator.optimizer == 'SGD':
        sim_optim_conf = [
            'SGD',
            opt.vdm.simulator.optimconf.lr,
            opt.vdm.simulator.optimconf.momentum,
            opt.vdm.simulator.optimconf.weight_decay,
            opt.vdm.simulator.optimconf.dampening,
            opt.vdm.simulator.optimconf.nesterov
        ]
    elif opt.vdm.simulator.optimizer == 'Adam':
        sim_optim_conf = [
            'Adam',
            opt.vdm.simulator.optimconf.lr,
            opt.vdm.simulator.optimconf.betas,
            opt.vdm.simulator.optimconf.weight_decay,
            opt.vdm.simulator.optimconf.eps,
            opt.vdm.simulator.optimconf.amsgrad
        ]
    elif opt.vdm.simulator.optimizer == 'AdamW':
        sim_optim_conf = [
            'AdamW',
            opt.vdm.simulator.optimconf.lr,
            opt.vdm.simulator.optimconf.betas,
            opt.vdm.simulator.optimconf.weight_decay,
            opt.vdm.simulator.optimconf.eps,
            opt.vdm.simulator.optimconf.amsgrad
        ]

    if opt.vdm.simulator.scheduler != 'None':
        if opt.vdm.simulator.scheduler == 'stepwise':
            sim_scheduler = lambda step, initial_lr: stepwiseDecaySheduler(step, initial_lr, gamma=0.001, power=0.75)
        elif opt.vdm.simulator.scheduler == 'inverse':
            sim_scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    else:
        sim_scheduler = None

    with open(os.path.join(log_dir, f'best_real_seen_classifier-{exp_name}.pkl'), 'rb') as f:
        seen_clf_params = torch.load(f)
    seen_clf = net.Classifier(opt.dataset.image_embedding_dim, nseenclasses, opt.classifier.metric, None, opt.classifier.weight_activation)
    seen_clf.load_state_dict(seen_clf_params['classifier'])

    simulator_agent = agent.SimAgent(opt.vdm.simulator.epochs, opt.vdm.feature_activation, seen_clf, nseenclasses, opt.dataset.image_embedding_dim, opt.vdm.simulator.nsample, sim_optim_conf, device,
                                     temperature=opt.classifier.temperature, scheduler=sim_scheduler, proto=proto, concentration=concentration, regularizer=None, blackbox=opt.vdm.simulator.blackbox)
    simulator_agent.fit()
    return simulator_agent


def simulate_virtual_data(simulator_agent, num, activation=None):
    print('simulate virtual data')
    features = []
    labels = []
    simulator = simulator_agent.simulator
    simulator.eval()

    with torch.no_grad():
        prototype, concentration = simulator.forward()
        print(concentration)
        print('concentration computed from text features')


        distribution = VonMisesFisher(prototype, concentration)

        for i in range(num):
            feature = distribution.rsample()
            if activation is not None:
                feature = activation(feature)
            feature = feature.cpu()
            label = torch.from_numpy(seenclasses).long()

            features.append(feature)
            labels.append(label)

    prototype = prototype.cpu()
    x = torch.cat(features, dim=0).cpu()
    y = torch.cat(labels, dim=0).cpu()

    simulator.train()

    return prototype, x, y


def train_generator(conditions):
    print('training the generator')
    if opt.vdm.generator.optimizer == 'SGD':
        gen_optim_conf = [
            'SGD',
            opt.vdm.generator.optimconf.lr,
            opt.vdm.generator.optimconf.momentum,
            opt.vdm.generator.optimconf.weight_decay,
            opt.vdm.generator.optimconf.dampening,
            opt.vdm.generator.optimconf.nesterov
        ]
    elif opt.vdm.generator.optimizer == 'Adam':
        gen_optim_conf = [
            'Adam',
            opt.vdm.generator.optimconf.lr,
            opt.vdm.generator.optimconf.betas,
            opt.vdm.generator.optimconf.weight_decay,
            opt.vdm.generator.optimconf.eps,
            opt.vdm.generator.optimconf.amsgrad
        ]

    with open(os.path.join(log_dir, f'best_real_seen_classifier-{exp_name}.pkl'), 'rb') as f:
        seen_clf_params = torch.load(f)
    seen_clf = net.Classifier(opt.dataset.image_embedding_dim, nseenclasses, opt.classifier.metric, None, opt.classifier.weight_activation)
    seen_clf.load_state_dict(seen_clf_params['classifier'])

    generator_agent = agent.GenAgent(opt.vdm.generator.epochs,  opt.vdm.feature_activation, seen_clf,
                                     opt.dataset.class_embedding_dim, opt.vdm.generator.noise_dim, opt.vdm.generator.hidden_dim, opt.dataset.image_embedding_dim,
                                     opt.vdm.generator.nsample, gen_optim_conf, device, conditions, opt.classifier.metric,
                                     opt.vdm.generator.gen_epochs, opt.vdm.generator.disdill_epochs, opt.classifier.temperature, opt.vdm.generator.blackbox)

    generator_agent.fit()
    return generator_agent


def generate_virtual_data(generator_agent, num, activation=None):
    print('generate virtual data')
    features = []
    labels = []
    generator = generator_agent.generator
    generator.eval()

    with torch.no_grad():
        for c in range(nseenclasses):
            cid = seenclasses[c]
            condition = generator_agent.conditions[c].unsqueeze(0).repeat(num, 1)
            noise = torch.randn(num, generator_agent.noise_dim)
            condition = condition.to(generator_agent.device)
            noise = noise.to(generator_agent.device)

            feature = generator(condition, noise)
            if activation is not None:
                feature = activation(feature)
            feature = feature.cpu()
            label = torch.from_numpy(np.ones(num) * (cid)).long()
            features.append(feature)
            labels.append(label)

    prototype = generator_agent.conditions.cpu()
    x = torch.cat(features, dim=0).cpu()
    y = torch.cat(labels, dim=0).cpu()

    generator.train()

    return prototype, x, y


def select_trusted_sample(x, y):
    with open(os.path.join(log_dir, f'best_real_seen_classifier-{exp_name}.pkl'), 'rb') as f:
        seen_clf_params = torch.load(f)
    seen_clf = net.Classifier(opt.dataset.image_embedding_dim, nseenclasses, opt.classifier.metric, None, opt.classifier.weight_activation)
    seen_clf.load_state_dict(seen_clf_params['classifier'])
    seen_clf = seen_clf.to(device)
    seen_clf.eval()

    with torch.no_grad():
        x = x.to(device)
        logits = seen_clf(x)
        preds = logits.topk(1, 1, True, True)[1].t().squeeze().cpu()
        mapped_y = map_label(y, torch.from_numpy(seenclasses).long())
        idx = (preds == mapped_y)
        x = x.cpu()
    return x[idx], y[idx]


def main():
    if opt.vdm.generation == 'gmm':
        # prototype = get_seen_prototype(opt.vdm.prototype, activation_dict[opt.vdm.prototype_activation])
        prototype = F.normalize(torch.from_numpy(cls_features[allclasses]).float())
        proto = prototype[:nseenclasses]

        concentration = get_seen_concentration(prototype)
        concentration = concentration.unsqueeze(1).max(dim=0, keepdim=True)[0].repeat(nseenclasses, 1)
        concentration = concentration * opt.vdm.theta

        x, y = sample_virtual_data(proto, concentration, opt.vdm.num, activation_dict[opt.vdm.feature_activation])
        print(y.size(0))
        x, y = select_trusted_sample(x, y)
        print(y.size(0))
        assert y.unique().size(0) == nseenclasses, 'Part of classes lost.'

    elif opt.vdm.generation == 'simulator':
        # prototype = get_seen_prototype(opt.vdm.prototype, activation_dict[opt.vdm.prototype_activation])
        prototype = F.normalize(torch.from_numpy(cls_features[allclasses]).float())
        proto = prototype[:nseenclasses]

        concentration = get_seen_concentration(proto)
        concentration = concentration.unsqueeze(1).max(dim=0, keepdim=True)[0].repeat(nseenclasses, 1)
        concentration = concentration * opt.vdm.theta

        simulator_agent = train_simulator(proto, concentration)
        sim_params = {
            'simulator': simulator_agent.sim_params
        }
        with open(os.path.join(log_dir, f'{opt.vdm.generation}-{opt.vdm.simulator.optimizer}_{opt.vdm.feature_activation}Act.pkl'), 'wb') as f:
            torch.save(sim_params, f)

        simulator_agent.simulator.load_state_dict(sim_params['simulator'])

        proto, x, y = simulate_virtual_data(simulator_agent, opt.vdm.num, activation_dict[opt.vdm.feature_activation])
        print(y.size(0))
        x, y = select_trusted_sample(x, y)
        print(y.size(0))
        assert y.unique().size(0) == nseenclasses, 'Part of classes lost.'

        # evaluate learned prototypes upon test seen features
        ground_truth = map_label(torch.from_numpy(test_seen_label).long(), torch.from_numpy(seenclasses).long()).to(device)
        logits = torch.from_numpy(test_seen_feature).float().to(device) @ proto.t().to(device)
        _, prediction = torch.max(logits, 1)
        targetclasses = ground_truth.unique()
        acc_per_class = torch.FloatTensor(nseenclasses).fill_(0)
        for i in range(nseenclasses):
            idx = (ground_truth == targetclasses[i])
            acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
        print(f'Accuracy evaluated by learned prototypes: {acc_per_class.mean() * 100}%', )


    elif opt.vdm.generation == 'generator':
        prototype = F.normalize(torch.from_numpy(cls_features[allclasses]).float())
        proto = prototype[:nseenclasses]

        generator_agent = train_generator(proto)
        gen_params = {
            'generator': generator_agent.generator.state_dict(),
            'student': generator_agent.student.state_dict()
        }
        with open(os.path.join(log_dir, f'{opt.vdm.generation}-{opt.vdm.generator.optimizer}_{opt.vdm.feature_activation}Act.pkl'), 'wb') as f:
            torch.save(gen_params, f)

        proto, x, y = generate_virtual_data(generator_agent, opt.vdm.num, activation_dict[opt.vdm.feature_activation])
        print(y.size(0))
        x, y = select_trusted_sample(x, y)
        print(y.size(0))
        assert y.unique().size(0) == nseenclasses, 'Part of classes lost.'

    ### train a seen classifier with virtual data ###
    #== Train a customized server seen classifier ==#
    # print("# of training samples: ", x.size(0))
    # seen_clf = train_classifier(x, y, proto=None)
    # pdb.set_trace()
    # seen_clf_params = {
    #     'classifier': seen_clf.clf_dict,
    #     'accuracy': seen_clf.best_acc
    # }
    # with open(os.path.join(log_dir, f'best_virtual_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}+{opt.vdm.num}Num.pkl'), 'wb') as f:
    #     torch.save(seen_clf_params, f)
    # == Train a customized server seen classifier ==#

    #== Train the linear probe of CLIP ==#
    print("# of training samples: ", x.size(0))


    mapped_y = map_label(y, torch.from_numpy(seenclasses).long())
    linear_probe = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, fit_intercept=False)
    linear_probe.fit(x.numpy(), mapped_y.numpy())
    lp_weights = torch.from_numpy(linear_probe.coef_).float()
    seen_clf = net.Classifier(opt.dataset.image_embedding_dim, nseenclasses, opt.classifier.metric, lp_weights, opt.classifier.weight_activation)

    logits = F.normalize(torch.from_numpy(test_seen_feature).float()) @ lp_weights.t()
    _, prediction = torch.max(logits.data, 1)
    ground_truth = map_label(torch.from_numpy(test_seen_label).long(), torch.from_numpy(seenclasses).long())
    classes = ground_truth.unique()
    acc_per_class = torch.FloatTensor(classes.size(0)).fill_(0)
    for i in range(classes.size(0)):
        idx = (ground_truth == classes[i])
        acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
    acc = acc_per_class.mean()
    print('Linear Probe Seen Accuracy: ', acc)


    seen_clf_params = {
        'classifier': seen_clf.state_dict(),
        'accuracy': acc
    }
    with open(os.path.join(log_dir, f'best_virtual_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}+{opt.vdm.num}Num.pkl'), 'wb') as f:
        torch.save(seen_clf_params, f)
    print('Having saved classifier weights.')
    # == Train the linear probe of CLIP ==#

    x, y = x.numpy(), y.numpy() + 1
    save_name = opt.vdm.generation + '-' + opt.dataset.image_embedding + '.mat'
    save_features = np.vstack((x, test_seen_feature, test_unseen_feature))
    save_labels = np.hstack((y, test_seen_label + 1, test_unseen_label + 1))
    mat_backbone['features'] = save_features.T
    mat_backbone['labels'] = save_labels
    mat_backbone['prototypes'] = proto.numpy().T
    save_file = os.path.join(save_dir, save_name)
    sio.savemat(save_file, mat_backbone)

    save_name = opt.vdm.generation + '-' + opt.dataset.class_embedding + '_splits.mat'
    mat_splits['trainval_loc'] = np.arange(len(y)) + 1
    mat_splits['train_loc'] = np.arange(len(y)) + 1
    mat_splits['val_loc'] = np.arange(len(y)) + 1
    mat_splits['test_seen_loc'] = np.arange(len(test_seen_label)) + 1 + len(y)
    mat_splits['test_unseen_loc'] = np.arange(len(test_unseen_label)) + 1 + len(y) + len(test_seen_label)
    save_file = os.path.join(save_dir, save_name)
    sio.savemat(save_file, mat_splits)


if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
seed_everything(opt.cuda, opt.manual_seed)

if opt.cuda != -1:
    if not torch.cuda.is_available():
        device = 'cpu'
        print("WARNING: You do not have a CUDA device, so you should probably run without --cuda.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
        device = 'cuda'
else:
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cpu'
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

if __name__ == "__main__":
    exp_name = f'{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}'
    print(exp_name)
    log_dir = os.path.join(opt.dataset.root, 'models', 'vdm', opt.dataset.name, exp_name)
    save_dir = os.path.join(opt.dataset.root, 'datasets', opt.dataset.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # loading .mat file
    mat_filepath = os.path.join(opt.dataset.root, 'datasets', opt.dataset.name, opt.dataset.image_embedding + '.mat')
    mat_backbone = sio.loadmat(mat_filepath)
    features = mat_backbone['features'].T
    labels = mat_backbone['labels'].astype(int).squeeze() - 1

    mat_filepath = os.path.join(opt.dataset.root, 'datasets', opt.dataset.name, opt.dataset.class_embedding + '_splits.mat')
    mat_splits = sio.loadmat(mat_filepath)
    trainval_loc = mat_splits['trainval_loc'].squeeze() - 1
    train_loc = mat_splits['train_loc'].squeeze() - 1
    val_unseen_loc = mat_splits['val_loc'].squeeze() - 1
    test_seen_loc = mat_splits['test_seen_loc'].squeeze() - 1
    test_unseen_loc = mat_splits['test_unseen_loc'].squeeze() - 1

    cls_features = mat_splits['cls_features'].T
    # cls_features = mat_splits['att'].T

    # prepare image embedding
    train_seen_feature = features[trainval_loc]
    train_seen_label = labels[trainval_loc]

    test_seen_feature = features[test_seen_loc]
    test_seen_label = labels[test_seen_loc]

    test_unseen_feature = features[test_unseen_loc]
    test_unseen_label = labels[test_unseen_loc]

    # prepare seen/unseen splits information
    _allclasses_names = np.squeeze(mat_splits['allclasses_names'])
    allclasses_names = []
    for i in range(len(_allclasses_names)):
        allclasses_names.append(_allclasses_names[i][0])
    allclasses_names = np.array(allclasses_names)

    seenclasses = np.unique(train_seen_label)
    nseenclasses = len(seenclasses)
    unseenclasses = np.unique(test_unseen_label)
    nunseenclasses = len(unseenclasses)
    nclasses = nseenclasses + nunseenclasses
    allclasses = np.hstack((seenclasses, unseenclasses))

    main()
    print(opt.dataset)