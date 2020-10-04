import argparse
import random
import os
import datetime
import numpy as np
from PIL import Image

import torch
from torch.nn import BCEWithLogitsLoss
from HRNet import HRNet
from torch_module.utils import train_model
import cv2


def custom_loss(target, predict):
    '''
        Args:
            target:   모델이 검출 해야하는 값
            predict:  모델 예측 결과

        Returns:
            rmse loss 결과 값을 반환
    '''
    return torch.sqrt(torch.mean(torch.pow(target-predict, 2)))


def precision(target, predict):
    '''
        Args:
            target:   모델이 검출 해야하는 값
            predict:  모델 예측 결과

        Returns:
            precision 수치를 반환
        Raise:
            target 검출 수 가 0 인 경우 else 구문에서 * 0 을 통해 precision 값을 0으로 고정.
    '''
    with torch.no_grad():
        ones = torch.ones(target.shape).cuda()
        zeros = torch.zeros(target.shape).cuda()
        target_values = torch.where(target > 0.5, ones, zeros)
        predicted = torch.where(predict > 0.5, ones, zeros)
        predicted_count = torch.sum(predicted)

        true_values = torch.where(predicted == 1, target_values, zeros)
        true_counts = torch.sum(true_values)

    return true_counts/predicted_count if predicted_count != 0 else true_counts * 0


def recall(target, predict):
    '''
        Args:
            target:   모델이 검출 해야하는 값
            predict:  모델 예측 결과

        Returns:
            recall 수치를 반환
    '''
    with torch.no_grad():
        ones = torch.ones(target.shape).cuda()
        zeros = torch.zeros(target.shape).cuda()
        true_values = torch.where(target > 0.5, ones, zeros)
        predicted = torch.where(predict > 0.5, ones, zeros)

        true_counts = torch.sum(true_values)

        predicted = torch.where(true_values == 1, predicted, zeros)
        predicted_counts = torch.sum(predicted)

    return predicted_counts/true_counts


def get_arguments():
    '''
        프로그램 인자 값을 받는 함수
        Args:

        Returns:
            epoch:          인자 값으로 --epoch n 을 입력 시, n의 epoch 반환. 기본 500을 사용
            batch_size:     인자 값으로 --batch n 을 입력 시, n의 batch 크기를 가짐. 기본 16을 사용
            save_path:      인자 값으로 --save path 를 입력 시, path를 모델 학습 과정을 저장할 경로로 지정.
                            기본 함수 내의 save_path 변수 사용.
            pretrain:       인자 값으로 --pretrain path 를 입력 시, pretrain 된 모델을 불러와서 사용. 기본 값으로 None 지정.
                            모델 test시에 사용.
            is_test:        인자 값으로 --test 를 입력 시, 모델 테스트를 수행 이때 pretrain 인자를 반드시 넣어야 함. 기본 값으로 False 지정.
            data_path:      인자 값으로 --data path 입력 시, path를 데이터가 저장되어 있는 경로 인식.
                             기본 값은 함수 내의 data_set_path 변수를 사용.
            video_path:     인자 값으로 --video path 입력 시, 비디오 경로를 설정. 학습 결과물에 대한 확인 용도.

    '''
    save_path = 'E:\\dataset\\Droplet\\train'
    data_set_path = 'E:\\dataset\\Droplet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', '-s', nargs='+', help='save path', default=[save_path], dest='save_path')
    parser.add_argument('--epoch', '-e', nargs='+', help='epoch count', default=[500], dest='epoch', type=int)
    parser.add_argument('--batch', '-b', nargs='+', help='batch size', default=[16], dest='batch_size', type=int)
    parser.add_argument('--pretrain', '-p', nargs='+', help='pretrain model', default=[None], dest='pretrain')
    parser.add_argument('--data', '-d', nargs='+', help='data path', default=[data_set_path], dest='data_path')
    parser.add_argument('--video', '-v', nargs='+', help='video data path', default=[None], dest='video_path')
    parser.add_argument(
        '--test', default=False, action="store_true",
        help='test'
    )
    save_path = parser.parse_args().save_path
    epoch = parser.parse_args().epoch
    batch_size = parser.parse_args().batch_size
    pretrain = parser.parse_args().pretrain
    is_test = parser.parse_args().test
    data_path = parser.parse_args().data_path
    video_path = parser.parse_args().video_path

    return epoch[0], batch_size[0], save_path[0], pretrain[0], is_test, data_path[0], video_path[0]




def train_loader(conf):
    '''
        
        설정값을 기반으로 데이터셋에서 입력 이미지와 검출해야되는 레이블 값을 반환하는 함수.

        Args:
            conf: 로더에서 사용하는 설정 값들이 저장된 dictionary.
                batch:      학습 시의 batch 크기
                path:       학습 데이터의 경로      
                is_train:   학습용 로더인지 validation용인지 분별 변수. 데이터 셔플에서 사용됨.

        Returns:
            x, seg_list: batch 크기의 이미지와 segmentation을 반환.

    '''
    def read_img(path, shape, mode='RGB'):        
        '''
            이미지를 학습을 위한 형태로 읽고 변형 하는 함수
            학습 이미지의 경우 정규화를 시켜 반환.
            레이블 이미지의 경우 0~1의 값으로 변환 시켜 반환.(255로 나눔)
            Args:
                path:   이미지의 경로
                shape:  반환할 이미지의 크기
            Returns:
                img:    변환된 이미지
        '''
        img = Image.open(path).resize(shape)
        if mode == 'L':
            img = img.convert(mode)
        img = np.asarray(img)
        if shape[0] == 256:
            img = np.reshape(img, (-1,))
            img = img / np.linalg.norm(img)
            img = np.reshape(img, (shape[0], shape[1], -1))
        else:
            img = img/255.
        return img

    batch_size = conf['batch']
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

            # pytorch의 경우 channel fisrt order 이기때문에 뒤에 있는 channel을 앞으로 가져와야 됨.
            input_img = np.moveaxis(input_img, -1, 0)

            # label 이미지의 경우 1 channel 이라 shape 형태가 (w, h) 로만 있어서 앞에 1채널을 추가 해줘야 됨.
            label_img = np.expand_dims(label_img, 0)
            label_img = np.where(label_img>0.5, 1, 0)

            x.append(input_img)
            seg_list.append(label_img)
        x = np.asarray(x)
        seg_list = np.asarray(seg_list)

        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
        seg_list = torch.from_numpy(seg_list).type(torch.FloatTensor).cuda()

        yield x, seg_list


def checkpoint(model, acc, validate_acc):
    '''
        모델의 학습 정도를 보고 best를 갱신 할 경우 모델을 저장함.
        Args:
            model:          학습중인 모델
            acc:            train acc 값
            validate_acc:   validate acc 값
        best 모델은 실행 경로에 생성.
    '''
    validate_acc[0] = validate_acc[0].detach().cpu().numpy()
    validate_acc[1] = validate_acc[1].detach().cpu().numpy()
    print('validate precision : ', validate_acc[0], ' max precision :',  checkpoint.max_acc[0])
    print('validate recall : ', validate_acc[1], ' max recall : ',  checkpoint.max_acc[1])
    print('validate mean : ', np.mean(validate_acc), ' max mean : ',  np.mean(checkpoint.max_acc))

    if np.mean(validate_acc) > np.mean(checkpoint.max_acc):
        print('*********saved new best model*********')
        checkpoint.max_acc[0] = validate_acc[0]
        checkpoint.max_acc[1] = validate_acc[1]
        torch.save(model.state_dict(), 'model{}.dict'.format(np.mean(checkpoint.max_acc)))
        traced_cell = torch.jit.trace(model.cuda(), torch.rand(1, 3, 256, 256).type(torch.FloatTensor).cuda())
        traced_cell.save('model{}.zip'.format(np.mean(checkpoint.max_acc)))


checkpoint.max_acc = np.asarray([0,0],  dtype=np.float)


def train_main(epoch, batch_size, path, save_path):
    '''
        학습의 기본 뼈대로 optimizer와 loss 함수, accuracy metrics를 지정하는 함수
        Args:
            epoch:          학습 반복 수
            batch_size:     학습 batch 크기
            path:           학습 데이터 경로
            save_path:      학습 과정 저장 경로
    '''

    net = HRNet(feature=128, depth=7, input_ch=3, output_ch=1, act='relu')

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = custom_loss

    train_model(epoch, net, loss, optim,
                {
                    'loader': train_loader,
                    'conf': {
                        'batch': batch_size,
                        'path': '{}\\train'.format(path),
                        'is_train': True,
                    }
                },
                {
                    'loader': train_loader,
                    'conf': {
                        'batch': batch_size,
                        'path': '{}\\validate'.format(path),
                        'is_train': False,
                    }
                },
                save_path='{}\\result'.format(save_path),
                tag=datetime.datetime.now(),
                accuracy=[
                    {'metrics': precision, 'name': 'precision'},
                    {'metrics': recall, 'name': 'recall'}
                ],
                checkpoint=checkpoint
                )


def test_main(pretrain_path, video_path):
    '''
        학습된 모델에 대하여 결과 영상 출력 함수.
        Args:
            pretrain_path:  pretrain 된 모델의 경로
            video_path:     결과 출력할 영상의 경로
        
        Raise:
            학습된 모델 경로와 확인할 영상의 경로를 입력 해야함.
        
    '''
    if pretrain_path is None or video_path is None:
        print("모델이나 영상의 경로가 지정되지 않았습니다")
        return
    net = HRNet(feature=128, depth=7, input_ch=3, output_ch=1, act='selu')
    net.load_state_dict(torch.load(pretrain_path))
    net.cuda()
    net.eval()

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('input', frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = rgb/255
            rgb = np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)
            rgb_t = torch.from_numpy(rgb).type(torch.FloatTensor).cuda()
            result = net(rgb_t).detach().cpu()
            rgb_n = result.numpy()
            rgb_n = np.squeeze(rgb_n, 0)
            rgb_n = np.transpose(rgb_n, (1, 2, 0))
            rgb_n = np.where(rgb_n > 0.5, 1, 0)
            rgb_n = np.asarray(rgb_n*255, np.uint8)

            edge = cv2.Canny(rgb_n, 100, 200)

            cv2.imshow('result', rgb_n)
            cv2.imshow('edge', edge)
            cv2.waitKey(10)
            del result


if __name__ == "__main__":
    epoch, batch_size, save_path, pretrain, is_test, data_set_path, video_path = get_arguments()
    if is_test:
        test_main(pretrain, video_path)
    else:
        train_main(epoch, batch_size, data_set_path, save_path)

