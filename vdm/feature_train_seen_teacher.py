import pdb
import sys,os

import torch

d = os.path.dirname(__file__)
parent_path = os.path.dirname(os.path.abspath(d))
sys.path.append(parent_path)
from PIL import Image
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    from torchvision.utils import save_image

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
cudnn.deterministic = True
cudnn.benchmark = True
from easydl import *
from sklearn.linear_model import LogisticRegression

from yaml_config import *
from dataset import *
from lr_schedule import *
import agent
import net


def main():
    # prepare datasets and dataloaders
    train_seen_set = FeatureSet(opt, mode='train')
    label_list = list(train_seen_set.labels.numpy())
    label_freq = Counter(label_list)
    sample_weight = {x: 1.0 / label_freq[x] if opt.dataset.resample else 1.0 for x in label_freq}
    sample_weights = [sample_weight[x] for x in label_list]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(label_list))
    train_seen_loader = torch.utils.data.DataLoader(dataset=train_seen_set,batch_size=opt.train.batch_size, sampler=sampler,
                                                    num_workers=opt.dataset.workers, drop_last=True)

    test_seen_set = FeatureSet(opt, mode='seen')
    test_seen_loader = torch.utils.data.DataLoader(dataset=test_seen_set, batch_size=opt.test.batch_size, shuffle=False,
                                                   num_workers=opt.dataset.workers, drop_last=False)

    #== Train a customized server seen classifier ==#
    # if opt.classifier.optimizer == 'SGD':
    #     clf_optim_conf = [
    #         'SGD',
    #         opt.classifier.optimconf.lr,
    #         opt.classifier.optimconf.momentum,
    #         opt.classifier.optimconf.weight_decay,
    #         opt.classifier.optimconf.dampening,
    #         opt.classifier.optimconf.nesterov
    #     ]
    # elif opt.classifier.optimizer == 'Adam':
    #     clf_optim_conf = [
    #         'Adam',
    #         opt.classifier.optimconf.lr,
    #         opt.classifier.optimconf.betas,
    #         opt.classifier.optimconf.weight_decay,
    #         opt.classifier.optimconf.eps,
    #         opt.classifier.optimconf.amsgrad
    #     ]
    # elif opt.classifier.optimizer == 'AdamW':
    #     clf_optim_conf = [
    #         'AdamW',
    #         opt.classifier.optimconf.lr,
    #         opt.classifier.optimconf.betas,
    #         opt.classifier.optimconf.weight_decay,
    #         opt.classifier.optimconf.eps,
    #         opt.classifier.optimconf.amsgrad
    #     ]
    #
    # if opt.classifier.scheduler != 'None':
    #     if opt.classifier.scheduler == 'stepwise':
    #         scheduler = lambda step, initial_lr: stepwiseDecaySheduler(step, initial_lr, gamma=0.001, power=0.75)
    #     elif opt.classifier.scheduler == 'inverse':
    #         scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75,
    #                                                                   max_iter=10000)
    # else:
    #     scheduler = None
    #
    # nclass = train_seen_set.nseenclasses
    # proto = None
    # # proto = torch.from_numpy(train_seen_set.cls_features[:nclass, :]).float()
    # # proto = torch.FloatTensor(train_seen_set.nseenclasses, opt.dataset.image_embedding_dim)
    # # for c in range(train_seen_set.nseenclasses):
    # #     this_train_seen_feature = train_seen_set.features[train_seen_set.labels == c]
    # #     this_train_seen_feature = this_train_seen_feature.mean(dim=0)
    # #     proto[c] = this_train_seen_feature
    #
    # seen_clf = agent.ClfAgent(opt, train_seen_loader, test_seen_loader, nclass, clf_optim_conf, device, proto=proto, scheduler=scheduler, input_img=False)
    # seen_clf.fit()
    #
    # pdb.set_trace()
    # seen_clf_params = {
    #     'classifier': seen_clf.clf_dict,
    #     'accuracy': seen_clf.best_acc
    # }
    # with open(os.path.join(log_dir, f'best_real_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}.pkl'), 'wb') as f:
    #     torch.save(seen_clf_params, f)
    #     print('Having saved classifier weights.')
    #== Train a customized server seen classifier ==#

    #== Train the linear probe of CLIP ==#
    linear_probe = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, fit_intercept=False)
    linear_probe.fit(train_seen_set.features.numpy(), train_seen_set.labels.numpy())
    lp_weights = torch.from_numpy(linear_probe.coef_).float()
    seen_clf = net.Classifier(opt.dataset.image_embedding_dim, train_seen_set.nseenclasses, opt.classifier.metric, lp_weights, opt.classifier.weight_activation)

    logits = test_seen_set.features @ lp_weights.t()
    _, prediction = torch.max(logits.data, 1)
    ground_truth = test_seen_set.labels
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


    with open(os.path.join(log_dir, f'best_real_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}.pkl'), 'wb') as f:
        torch.save(seen_clf_params, f)
    print('Having saved classifier weights.')

    #== Train the linear probe of CLIP ==#

    #== Use Centers as the Classifier ==#
    # proto = []
    # concentration = []
    # for c in range(train_seen_set.nseenclasses):
    #     idx = (train_seen_set.labels == c)
    #     # this_features = F.normalize(train_seen_set.features[idx])
    #     # cos = this_features @ this_features.t()
    #     # std = torch.arccos(cos.min()) / 8
    #     # concentration.append(1 / std ** 2)
    #     proto.append(F.normalize(train_seen_set.features[idx]).mean(dim=0))
    # # print(concentration)
    # proto = torch.stack(proto, dim=0)
    # seen_clf = net.Classifier(opt.dataset.image_embedding_dim, train_seen_set.nseenclasses, opt.classifier.metric,
    #                           proto, opt.classifier.weight_activation)
    #
    # logits = F.normalize(test_seen_set.features) @ F.normalize(proto).t()
    # _, prediction = torch.max(logits.data, 1)
    # ground_truth = test_seen_set.labels
    # classes = ground_truth.unique()
    # acc_per_class = torch.FloatTensor(classes.size(0)).fill_(0)
    # for i in range(classes.size(0)):
    #     idx = (ground_truth == classes[i])
    #     acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
    # acc = acc_per_class.mean()
    # print('Centers as Classifier Seen Accuracy: ', acc)
    #
    # pdb.set_trace()
    # seen_clf_params = {
    #     'classifier': seen_clf.state_dict(),
    #     'accuracy': acc
    # }
    # with open(os.path.join(log_dir, f'best_real_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}.pkl'), 'wb') as f:
    #     torch.save(seen_clf_params, f)
    # print('Having saved classifier weights.')
    # == Use Centers as the Classifier ==#

    # seen_confusion_matrix = seen_clf.confusion_matrix.numpy()
    # df_seen_confusion_matrix_scalar = pd.DataFrame(
    #     data=seen_confusion_matrix,
    #     index=data.seenclasses_names,
    #     columns=data.seenclasses_names
    # )
    # df_seen_confusion_matrix_perception = pd.DataFrame(
    #     data=seen_confusion_matrix / np.sum(seen_confusion_matrix, axis=1, keepdims=True),
    #     index=data.seenclasses_names,
    #     columns=data.seenclasses_names
    # )
    # saveXlsxTable(log_dir, 'seen_cosconfusion_matrix',
    #               [df_seen_confusion_matrix_scalar, df_seen_confusion_matrix_perception],
    #               ['scalar', 'perception'])

if __name__ == "__main__":
    exp_name = f'{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}'
    print(exp_name)
    log_dir = os.path.join(opt.dataset.root, 'models', 'vdm', opt.dataset.name, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    main()





