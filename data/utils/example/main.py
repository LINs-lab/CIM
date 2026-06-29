import time
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms

from data.utils.load_model import load_model
from data.utils.load_dataset import load_dataset, load_normalize, load_denormalize


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, val_loader, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg) +
        "Top-1 acc = {:.6f},\t".format(top1.avg) +
        #    "Top-1 err = {:.6f},\t".format(100 - top1.avg) +
        "Top-5 err = {:.6f},\t".format(100 - top5.avg) +
        "val_time = {:.6f}".format(time.time() - t1))
    print(logInfo)

    return top1.avg


if __name__ == "__main__":

    # dataset = 'imagenet-1k'
    # model = 'efficientnet_b0'

    # normalize = load_normalize(dataset=dataset)
    # denormalize = load_denormalize(dataset=dataset)

    # test_dataset = load_dataset(dataset=dataset, train=False)

    # model = load_model(model_name=model,
    #                    dataset=dataset,
    #                    pretrained=True,)

    # model = model.cuda()

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=256,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=False,
    # )

    # _ = validate(model, test_loader, epoch=0)

