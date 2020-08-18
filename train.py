import argparse
import random
import os
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from HRNet import HRNet
from torch_module.utils import train_model
import cv2


def image_blend(org, base):
    ary = np.array(base, dtype=np.float)
    # ary /= 255.
    plt.imsave('temp.png',ary, cmap='hot')
    temp = Image.open('temp.png')
    temp = temp.resize((256,256))
    temp = temp.convert("RGBA")
    t = org.convert("RGBA")
    blended = Image.blend(t, temp, 0.5)
    return blended


def acc(target, predict):
    with torch.no_grad():
        mean = torch.mean(torch.abs((target-predict)))
    return mean


def get_arguments():
    data_set_path = 'E:\\dataset\\skeleton\\train'
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', '-s', nargs='+', help='save path', default=[data_set_path], dest='save_path')
    parser.add_argument('--epoch', '-e', nargs='+', help='epoch count', default=[500], dest='epoch', type=int)
    parser.add_argument('--batch', '-b', nargs='+', help='batch size', default=[16], dest='batch_size', type=int)
    parser.add_argument('--pretrain', '-p', nargs='+', help='pretrain model', default=[None], dest='pretrain')
    parser.add_argument(
        '--test', default=False, action="store_true",
        help='test'
    )
    save_path = parser.parse_args().save_path
    epoch = parser.parse_args().epoch
    batch_size = parser.parse_args().batch_size
    pretrain = parser.parse_args().pretrain
    is_test = parser.parse_args().test

    return epoch[0], batch_size[0], save_path[0], pretrain[0], is_test


def custom_loss(target, predict):
    return torch.mean(torch.pow(torch.abs(target-predict), 2))


def train_loader(conf):
    def read_img(path, shape, mode='RGB'):
        img = Image.open(path).resize(shape)
        if mode == 'L':
            img = img.convert(mode)
        img = np.asarray(img)
        return img/255.

    batch_size = conf['batch_size']
    data_path = conf['path']

    input_path = "{}\\input".format(data_path)
    label_path = "{}\\label".format(data_path)

    file_names = os.listdir(input_path)

    iter_len = len(file_names)//batch_size

    if conf['is_train']:
        random.shuffle(file_names)

    for iter in range(iter_len):
        x = []
        seg_list = []
        for batch in range(batch_size):
            b_idx = batch_size * iter + batch
            input_img = read_img(input_path+'\\{}'.format(file_names[b_idx]), (256,256))
            label_img = read_img(label_path+'\\{}'.format(file_names[b_idx]), (64,64), 'L')

            input_img = np.moveaxis(input_img, -1, 0)
            label_img = np.expand_dims(label_img, -1)
            label_img = np.moveaxis(label_img, -1, 0)

            x.append(input_img)
            seg_list.append(label_img)
        x = np.asarray(x)
        seg_list = np.asarray(seg_list)

        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
        seg_list = torch.from_numpy(seg_list).type(torch.FloatTensor).cuda()

        yield x, seg_list


def checkpoint(model, loss, validate_loss):
    print(validate_loss.item(), checkpoint.min_loss)
    if validate_loss.item() < checkpoint.min_loss:
        checkpoint.min_loss = validate_loss.item()
        torch.save(model.state_dict(), 'model{}.dict'.format(validate_loss.item()))


checkpoint.min_loss = 100


def train_main(epoch, batch_size):
    net = HRNet(feature=128, depth=7, input_ch=3, output_ch=1, act='selu')
    # net.info()

    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_model(epoch, net, custom_loss, optim,
                {
                    'loader': train_loader,
                    'conf': {
                        'batch_size': batch_size,
                        'path': 'E:\\dataset\\droplet\\train',
                        'is_train': True,
                    }
                },
                {
                    'loader': train_loader,
                    'conf': {
                        'batch_size': batch_size,
                        'path': 'E:\\dataset\\droplet\\validate',
                        'is_train': False,
                    }
                },
                save_path='E:\\dataset\\droplet\\result',
                tag=datetime.datetime.now(),
                accuracy=[
                    {'metrics':acc, 'name':'mae'}
                ],
                checkpoint=checkpoint)


def test_main(save_path):
    i = 0
    net = HRNet(feature=128, depth=7, input_ch=3, output_ch=1, act='selu')
    net.load_state_dict(torch.load('model0.011182134971022606.dict'))
    net.cuda()
    net.eval()
    for x, y in train_loader({
        'batch_size': 1,
        'path': 'E:\\dataset\\droplet\\validate',
        'is_train': False,
        'repeat': repeat
    }):
        result = net(x)
        seg = y.cpu().detach().numpy()
        input_img = x[0].cpu().numpy()
        seg_img = seg[0]

        input_img = np.moveaxis(input_img, 0, -1)
        input_img = np.array(input_img*255, dtype=np.uint8)
        input_img = cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR)

        show = cv2.resize(input_img, (512, 512))

        cv2.imshow("input", show)

        seg_img = np.array(np.reshape(np.where(seg_img>0.5, 1, 0), (64, 64, 1))*255,dtype=np.uint8)
        contours, hierarchy = cv2.findContours(seg_img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        base = input_img.copy()
        for cnt in contours:
            cnt = np.array(cnt)*4
            cv2.drawContours(base, [cnt], 0, (0,0,255),1)
        print(base.shape)
        base = cv2.resize(base, (512, 512))
        cv2.imshow("result", base)

        seg_img = np.tile(seg_img, 3)
        seg_img = cv2.resize(seg_img, (256,256))
        target_img = np.array(input_img & seg_img, dtype=np.uint8)
        show = cv2.resize(target_img, (512, 512))
        cv2.imshow("target", show)
        cv2.waitKey(10)

        with torch.no_grad():
            seg_img_pred = result[-1][0].cpu().numpy()
            seg_img_pred = np.reshape(seg_img_pred, (64, 64))
            seg_img = np.array(np.reshape(np.where(seg_img_pred > 0.5, 1, 0), (64, 64, 1)) * 255, dtype=np.uint8)
            seg_img = np.array(np.reshape(np.where(seg_img > 0.5, 1, 0), (64, 64, 1)) * 255, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(seg_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            base = input_img.copy()
            for cnt in contours:
                cnt = np.array(cnt) * 4
                cv2.drawContours(base, [cnt], 0, (0, 0, 255), 1)
            print(base.shape)
            base = cv2.resize(base, (512, 512))
            cv2.imshow("result_pred", base)

            seg_img = np.tile(seg_img, 3)
            seg_img[:, :,  1] = 0
            seg_img[:, :, 2] = 0
            cv2.imshow("predict", show)
            cv2.waitKey(0)

        plt.show()
        plt.close()

        i = i + 1


if __name__ == "__main__":
    epoch, batch_size, repeat, n_stack, save_path, pretrain, is_test = get_arguments()
    if is_test:
        test_main(save_path)

    else:
        train_main(300, batch_size)

