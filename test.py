from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wideresnet as models
import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
from util import *

from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=50,
                        help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=375,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# from IPython import embed
# embed()
use_cuda = torch.cuda.is_available()
if use_cuda:
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),#不转换为PIL会报错
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.ToPILImage(),  # 不转换为PIL会报错
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print('==> Preparing data')
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_CUB200('./data/train', args.n_labeled, transform_train=transform_train, transform_val=transform_val)
    # embed()
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # embed()
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("==> creating model PMG")

    ### 加载模型model
    def create_model(ema=False):
        model = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
        if use_cuda:
            model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    modelp = torch.nn.DataParallel(model, device_ids=[0])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # loss, loss1, loss2, loss3 = train(labeled_trainloader, unlabeled_trainloader, model, ema_optimizer, train_criterion, start_epoch, use_cuda)
    # print("===>Training ended")
    # print("loss_avg:",loss)
    # print("loss1_avg:", loss1)
    # print("loss2_avg:", loss2)
    # print("loss3_avg:", loss3)

    print("===>Start testing")
    model_path = "./output/model_5_2.pth"
    trained_model = torch.load(model_path,map_location = torch.device('cpu'))
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    test_acc,_,_ = validate(test_loader, trained_model, criterion, use_cuda)
    # print("test_loss:", test_loss)
    print("test_acc:",test_acc)


def train(labeled_trainloader, unlabeled_trainloader, model, ema_optimizer, criterion, start_epoch, use_cuda):
    bar = Bar('Training', max=args.train_iteration)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    losses = AverageMeter()

    end = time.time()


    ### optimizer setting
    optimizer = optim.SGD([
        {'params': model.classifier_concat.parameters(), 'lr': 0.002},
        {'params': model.conv_block1.parameters(), 'lr': 0.002},
        {'params': model.classifier1.parameters(), 'lr': 0.002},
        {'params': model.conv_block2.parameters(), 'lr': 0.002},
        {'params': model.classifier2.parameters(), 'lr': 0.002},
        {'params': model.conv_block3.parameters(), 'lr': 0.002},
        {'params': model.classifier3.parameters(), 'lr': 0.002},
        {'params': model.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))


        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        model.train()
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
                # embed()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()

            try:
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()  ### 问题：这里为什么是(inputs_u, inputs_u2),_？
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                # embed()
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
                # inputs_u, inputs_u2 = unlabeled_train_iter.next()

            # measure data loading time
            data_time.update(time.time() - end)

            batch_size = inputs_x.size(0)

            # embed()
            # Transform label to one-hot
            # targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)
            targets_x = torch.zeros(batch_size, 200).scatter_(1, targets_x.unsqueeze(1).long(), 1)

            if use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                # embed()
                # inputs_u = inputs_u.resize_(64, 3, 512, 512)
                # outputs_u = model(inputs_u)
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                # embed()
                shape1 = outputs_u[0].shape[0]
                shape2 = outputs_u[0].shape[1]
                if use_cuda:
                    p = torch.zeros(shape1, shape2).cuda()
                else:
                    p = torch.zeros(shape1, shape2)

                ### 可能后边logits出现的所有的问题都出在这
                for i in range(len(outputs_u)):
                    p = p + (torch.softmax(outputs_u[i], dim=1) + torch.softmax(outputs_u2[i], dim=1)) / 2
                # p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            # embed()
            ### inputs_x inputs_u inputs_u2进行拼接
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(args.alpha, args.alpha)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            ### input_a input_b是啥？
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            ### 这一个interleave没问题
            mixed_input = interleave(mixed_input, batch_size)

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])

            ###### input1
            optimizer.zero_grad()
            ### get mixed_input1
            mixed_input1 = jigsaw_generator(mixed_input, 8)
            logits = get_logits(0, model, mixed_input1)
            loss1 = get_loss(logits, criterion, mixed_target, batch_size, batch_idx, epoch)
            # record loss
            losses1.update(loss1.item(), inputs_x.size(0))
            loss1.backward()
            optimizer.step()
            ema_optimizer.step()

            ###### input2
            optimizer.zero_grad()
            ### get mixed_input2
            mixed_input2 = jigsaw_generator(mixed_input, 4)
            logits = get_logits(1, model, mixed_input2)
            loss2 = get_loss(logits, criterion, mixed_target, batch_size, batch_idx, epoch)
            # record loss
            losses2.update(loss2.item(), inputs_x.size(0))
            loss2.backward()
            optimizer.step()
            ema_optimizer.step()

            ###### input3
            optimizer.zero_grad()
            ### get mixed_input3
            mixed_input3 = jigsaw_generator(mixed_input, 2)
            logits = get_logits(2, model, mixed_input3)
            loss3 = get_loss(logits, criterion, mixed_target, batch_size, batch_idx, epoch)
            # record loss
            losses3.update(loss3.item(), inputs_x.size(0))
            loss3.backward()
            optimizer.step()
            ema_optimizer.step()

            ###### total input
            optimizer.zero_grad()
            logits = get_logits(3, model, mixed_input)
            loss = get_loss(logits, criterion, mixed_target, batch_size, batch_idx, epoch)
            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}|{size}|{epoch}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_1: {loss1:.4f} | Loss_2: {loss2:.4f} | Loss_3: {loss3:.4f}'.format(
                batch=batch_idx + 1,
                size=args.train_iteration,
                epoch=epoch + 1,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                loss3=losses3.avg,
                # loss_x=losses_x.avg,
                # loss_u=losses_u.avg,
                # w=ws.avg,
            )
            bar.next()
    bar.finish()

    torch.save(model, "./output/model_5.pth")
    return (losses.avg, losses1.avg, losses2.avg, losses3.avg)

def validate(testloader, net, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()


    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")


    for batch_idx, (inputs, targets) in enumerate(testloader):
        # print("batch_idx",batch_idx)
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat= net(inputs)
        ### 预测结果为output_concat
        outputs_com = output_1 + output_2 + output_3 + output_concat

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss

    # end = time.time()
    # bar = Bar('{mode}', max=len(valloader))
    # with torch.no_grad():
    #     embed()
    #     for batch_idx, (inputs, targets) in enumerate(valloader):
    #         # measure data loading time
    #         data_time.update(time.time() - end)
    #
    #         if use_cuda:
    #             inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #         # compute output
    #         outputs = model(inputs)
    #         # embed()
    #         loss = criterion(outputs, targets)
    #
    #         # measure accuracy and record loss
    #         prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
    #         losses.update(loss.item(), inputs.size(0))
    #         top1.update(prec1.item(), inputs.size(0))
    #         top5.update(prec5.item(), inputs.size(0))
    #
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #
    #         # plot progress
    #         bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                     batch=batch_idx + 1,
    #                     size=len(valloader),
    #                     data=data_time.avg,
    #                     bt=batch_time.avg,
    #                     total=bar.elapsed_td,
    #                     eta=bar.eta_td,
    #                     loss=losses.avg,
    #                     top1=top1.avg,
    #                     top5=top5.avg,
    #                     )
    #         bar.next()
    #     bar.finish()
    # return (losses.avg, top1.avg)

### get_mixed_input
def jigsaw_generator(images, n):
    result = []
    for image in images:
        l = []
        for a in range(n):
            for b in range(n):
                l.append([a, b])
        block_size = 448 // n
        rounds = n ** 2
        random.shuffle(l)
        jigsaws = image.clone()
        for i in range(rounds):
            x, y = l[i]
            temp = jigsaws[..., 0:block_size, 0:block_size].clone()
            jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                       y * block_size:(y + 1) * block_size].clone()
            jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
        result.append(jigsaws)
    return result



def get_logits(input_id,model,mixed_input):
    logits = []
    # embed()
    logits.append(model(mixed_input[0])[input_id])
    ### model(mixed_input[0])[3]相当于_, _, _, logit = model(mixed_input[0])
    ### targets_u
    for input in mixed_input[1:]:
        logit = model(input)[input_id]
        logits.append(logit)  ### 特征提取

    return logits

def get_loss(logits,criterion,mixed_target,batch_size,batch_idx,epoch):

    # logits = []
    # # embed()
    # logits.append(model(mixed_input[0])[3])
    # ### model(mixed_input[0])[3]相当于_, _, _, logit = model(mixed_input[0])
    # ### targets_u
    # for input in mixed_input[1:]:
    #     _, _, _, logit = model(input)
    #     logits.append(logit)  ### 特征提取

    ### 此时logits[i].shape [2,200]
    # put interleaved samples back
    ### 问题出在这个interleave上
    # embed()
    logits = interleave(logits, batch_size)
    # embed()
    ### 原来的代码
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    ### 这里是有问题的
    # logits_x = logits[0][:batch_size]# 有问题 应该是logits_x = logits[0]
    # logits_u = torch.cat(logits[batch_size:], dim=0)# 有问题 应该是logits_u = torch.cat(logits[1:], dim=0)

    ### 计算半监督损失
    ### logits_x logits_u就是outputs_x outputs_u
    Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                          epoch + batch_idx / args.train_iteration)

    loss = Lx + w * Lu

    return loss

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
### 半监督损失 Lx + Lu + w
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        ### 有标签数据的损失 交叉熵损失
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        ### 无标签数据的损失 计算l2 loss
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    # embed()
    return [torch.cat(v, dim=0) for v in xy]

def interleave_(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    # embed()
    total_tensors = []
    for v in xy:
        tensors = []
        for vv in v:
            if len(vv) > 0:
                tensors.append(vv[0])# vv[0]是[2,200]所以cat之后就是[4,200]
        total_tensors.append(tensors)
    return [torch.cat(v, dim=0) for v in total_tensors]

if __name__ == '__main__':
    main()