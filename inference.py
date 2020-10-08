import torch.jit
import json
import torch
import math
import cv2
import numpy as np


def convert_image_for_input(image):
    '''
        모델에 입력하기 위해 Tensor 타입으로 이미지를 변환하는 함수
        Args
            image: 입력 이미지로 크기를 (256,256)으로 변경하고 255로 나눈 값을 이용한다.
        Return
            image: tensor 타입의 이미지

    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = image / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor).cuda()

    return image


class TrackingInfo:
    '''
        트랙킹 정보 클래스
        die:    트랙킹 종료 설정
        checked: 트랙킹 시 해당 객체를 사용했는지 체크용
        center: 발생부터 종료까지 검출된 기포의 위치
        area: 발생부터 종료까지 검출된 기포의 영역 크기
    '''

    def __init__(self, contours, id, frame):
        self.die = False
        self.checked = False
        self.id = id
        self.center = []
        self.area = []
        self.frames = []
        self.avg = 0
        self.set_contour(contours, frame)

    def set_find(self, find):
        self.checked = find

    def set_die(self):
        self.die = True

    def set_contour(self, contours, frame):
        '''
            트랙킹 정보 업데이트 함수.
            입력 받은 contour 값을 이용해 중심 위치와 면적의 크기를 새로 갱신한다.
            Args:
                contours: 새로운 contour 값
        '''
        x1 = np.max(np.asarray(contours[:, 0, 0]))
        x2 = np.min(np.asarray(contours[:, 0, 0]))

        y1 = np.max(np.asarray(contours[:, 0, 1]))
        y2 = np.min(np.asarray(contours[:, 0, 1]))
        self.center.append(((x1 + x2) / 2., (y1 + y2) / 2.))
        self.frames.append(frame)
        if len(self.center) > 2:
            speed = [math.sqrt(
                (self.center[j][0] - self.center[j + 1][0]) ** 2 + (self.center[j][1] - self.center[j + 1][1]) ** 2) for
                j in range(len(self.center) - 1)]
            speed = np.array(speed, dtype=float)
            self.avg = np.mean(speed)
        self.base = cv2.drawContours(np.zeros((64, 64), np.uint8), [contours], 0, 255, -1)
        self.area.append(np.sum(self.base // 255))

    def update_info(self, contour, frame):
        self.set_contour(contour, frame)

    def __str__(self):
        info = 'id,{0},die,{1}'.format(self.id, self.die)
        info = info + ',frame,'
        for fr in self.frames:
            info = info + '{},'.format(fr)

        info = info + 'center,'
        for ct in self.center:
            info = info + '{},'.format(ct)

        info = info + 'area,'
        for ar in self.area:
            info = info + '{},'.format(ar)
        info = info[:-1]
        return info


class TrackingInfoMaker:

    def __init__(self):
        self.random_id = 0
        self.contours = {}

    def create_new_id(self):
        '''
            tracking info 를 만들기 위한 새로운 ID 생성
            Args
            Returns
                new ID


        '''
        return len(self.contours) + 1

    def create_info(self, contour, frame):
        '''
            새로운 트랙킹 객체 생성
            Args
                contour: 객체 생성에 필요한 contour 정보
            Returns
                새로 생성된 Tracking 객체
        '''
        new_id = self.create_new_id()
        self.contours[new_id] = TrackingInfo(contour, new_id, frame)
        return self.contours[new_id]

    def find_max_possible_contour(self, origin, contour_base, is_used):
        '''
            origin 정보와 contour_base간 교차영역을 찾아 가장 높은 Index를 반환
            Args
                origin: Tracking 정보가 Update 될 객체
                contour_base: Update하기위해 비교연산을 위한 contour 영역들
                is_used: contour 중 이미 다른 Tracking 객체 update에 사용된 객체를 표시. 해당 값이 True면 사용되지 않음.
            Return
                max_idx: 가장 교차 영역이 큰 index 반환.
        '''
        contour_count = len(contour_base)
        max_duplicate = 0
        max_idx = -1

        # Tracking 객체 업데이트에 적합한 contour 검색
        for idx in range(contour_count):
            if is_used[idx]:
                continue

            # contour가 가지는 값은 0, 255만 있어서 0, 1로 변경하여 서로 * 연산을하면 교차되는 영역만 픽셀이 1로 남게됨.
            # 이를 이용해 남은 1의 합을 구하면 교차영역의 넓이가 됨.
            duplicate = np.sum(contour_base[idx] // 255 * origin.base // 255)

            # if duplicate > max_duplicate and duplicate > self.contours[ct_key].area[-1]//2:
            if duplicate > max_duplicate:
                max_duplicate = duplicate
                max_idx = idx
        return max_idx

    def update_tracking_info(self, contours, frame):
        '''
            입력받은 예비 contour 값들을 기반으로 기존에 존재하던 Tracking 객체와 비교하여 정보를 갱신하고
            정보 갱신에 사용되지않은 contour는 새로운 객체로 생성
            Tracking 방식은 contour간 교차 영역이 가장 큰 객체간 정보를 업데이트 한다.
            Args
                contours: contour 정보들
            Return
        '''
        contour_count = len(contours)
        is_used = [False for _ in range(contour_count)]
        # Tracking에 필요한 contour 영역 생성
        base = np.zeros((64, 64), np.uint8)
        contour_base = [cv2.drawContours(base.copy(), contours, i, 255, -1) for i in range(contour_count)]
        for ct_key in self.contours:
            if self.contours[ct_key].die:
                continue
            else:
                self.contours[ct_key].set_find(False)
                max_idx = self.find_max_possible_contour(self.contours[ct_key], contour_base, is_used)

                if max_idx != -1:
                    is_used[max_idx] = True
                    self.contours[ct_key].update_info(contours[max_idx], frame)
                    self.contours[ct_key].set_find(True)
                else:
                    self.contours[ct_key].set_die()

        for idx in range(contour_count):
            if not is_used[idx]:
                self.create_info(contours[idx], frame)


def predict_to_image(predicted):
    '''
        예측된 Tensor를 이미지 배열로 변환하는 함수
        Args
            predicted: 예측된 Tensor 값
        Returns
            np_image: 변환된 이미지
    '''
    np_image = predicted.detach().cpu().numpy()
    np_image = np.squeeze(np_image)
    np_image = np.expand_dims(np_image, -1)
    np_image = np.where(np_image > 0.5, 255, 0)
    np_image = np.array(np_image, dtype=np.uint8)
    return np_image


def get_contour(image):
    '''
        image로 부터 contour 를 잡아냄.
        Args
            image: prediction된 이미지로 segment 이미지가 입력으로 사용됨
        Returns
            contours: cv2 findContour 함수를 사용해 찾아낸 contour를 반환.

    '''
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def inference(model_path, video_path):
    '''
        inference 방법에 대한 sample 함수
        TrackingInfoMaker 객체를 생성하고
        model_path를 이용해 모델을 불러온 뒤 영상을 계속해서 입력받아 정보를 갱신한다.
        Args
            model_path: 모델의 경로 확장자는 일반적으로 zip을 사용 pb도 괜찮음
            video_path: 액적 영상의 경로를 입력
    '''

    # model 로드
    model = torch.jit.load(model_path)
    print('model loaded')

    # model 쿠다 설정
    if torch.cuda.is_available():
        print('model to cuda')
        model.cuda()

    # 트래킹 정보 생성자 생성
    maker = TrackingInfoMaker()
    print('TrackingInfoMaker made')

    # 트래킹 영상 로드
    vc = cv2.VideoCapture(video_path)
    ret, image = vc.read()
    print('Start tracking')
    frame_num = 1
    while ret:
        frame_num = frame_num + 1
        image = convert_image_for_input(image)
        predict = model(image)

        predict = predict_to_image(predict)

        # contour 추출
        contours = get_contour(predict)

        cv2.imshow('result', predict)

        # Tracking 정보 갱신
        maker.update_tracking_info(contours, frame_num)

        # 입력 영상 출력을 위해 변환
        input_image = image.detach().cpu().numpy()
        input_image = np.squeeze(input_image)
        input_image = np.transpose(input_image, (1, 2, 0))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        input_image = np.array(input_image * 255, dtype=np.uint8)
        draw_image = input_image.copy()
        for ct_key in maker.contours:
            if maker.contours[ct_key].die:
                continue
            else:
                center = maker.contours[ct_key].center[-1]
                cv2.putText(img=draw_image, text='{}'.format(ct_key),
                            org=(int(center[0] * 4), int(center[1] * 4)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255),
                            lineType=2)
                resized_base = cv2.resize(maker.contours[ct_key].base, (256, 256))
                edge = cv2.Canny(resized_base, 128, 255)
                edge = np.expand_dims(edge, axis=-1)
                edge = np.ones((256, 256, 3), dtype=np.uint8) * edge
                draw_image = draw_image & (255 - edge)
                draw_image = draw_image | edge

        cv2.imshow('draw', draw_image)
        cv2.waitKey(10)
        ret, image = vc.read()
    print('End tacking')
    cv2.destroyAllWindows()
    return maker.contours


if __name__ == "__main__":
    model_path = 'E:\\dataset\\Droplet\\best_model.zip'
    video_path = 'E:\\dataset\\Droplet\\video\\g.avi'
    inference(model_path, video_path)
