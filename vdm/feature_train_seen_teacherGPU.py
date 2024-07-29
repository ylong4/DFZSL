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

import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionGPU(BaseEstimator, ClassifierMixin,nn.Module):
    def __init__(self, random_state=0, C=1.0, max_iter=100, verbose=0, fit_intercept=True, tol=1e-4):
        super().__init__()
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y):
        X_tensor = torch.from_numpy(X).float().to(torch.device('cuda'))
        y_tensor = torch.from_numpy(y).long().to(torch.device('cuda'))

        self.coef_ = torch.zeros((X.shape[1],))
        self.intercept_ = torch.zeros((1,))
        self.n_iter_ = 0


        n_samples = len(y)
        y=torch.from_numpy(y).long()
        class_weights = torch.tensor([n_samples / (2 * torch.sum(y == 0)), n_samples / (2 * torch.sum(y == 1))]).to(
            torch.device('cuda'))

        for i in range(self.max_iter):
            y_pred = torch.sigmoid(self.intercept_ + torch.matmul(X_tensor, self.coef_))
            loss = F.binary_cross_entropy(y_pred, y_tensor.float(), weight=class_weights, reduction='mean')

            if self.verbose > 0 and i % self.verbose == 0:
                print("Iteration %d, loss = %f" % (i, loss))

            if torch.abs(loss - self.loss_) < self.tol:
                break

            self.loss_ = loss
            self.n_iter_ += 1

            y_diff = y_tensor - y_pred
            self.intercept_ += self.C * torch.mean(y_diff)
            self.coef_ += self.C * torch.mean(X_tensor * (y_diff.view(-1, 1)), axis=0)

    def predict(self, X):
        y_pred_proba = torch.sigmoid(
            torch.matmul(torch.from_numpy(X).float().to(torch.device('cuda')), self.coef_) + self.intercept_)
        y_pred = (y_pred_proba >= 0.5).to(torch.device('cpu')).detach().numpy().astype(int)
        return y_pred


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


    linear_probe = LogisticRegressionGPU(random_state=0, C=0.316, max_iter=1000, verbose=1, fit_intercept=False)
    linear_probe = linear_probe.cuda()
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

    pdb.set_trace()
    seen_clf_params = {
        'classifier': seen_clf.state_dict(),
        'accuracy': acc
    }

    pdb.set_trace()
    # with open(os.path.join(log_dir, f'best_real_seen_classifier-{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}.pkl'), 'wb') as f:
    #     torch.save(seen_clf_params, f)
    # print('Having saved classifier weights.')


if __name__ == "__main__":
    exp_name = f'{opt.classifier.weight_activation}Act+{opt.classifier.metric}Metric_{opt.classifier.optimizer}'
    print(exp_name)
    log_dir = os.path.join(opt.dataset.root, 'models', 'vdm', opt.dataset.name, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    main()





