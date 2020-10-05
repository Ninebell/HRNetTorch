import torch.jit
import torch
import cv2
import numpy as np


class TrackingInfo:
    def __init__(self, contours, id):
        self.die = False
        self.checked=False
        self.id = id
        self.center = []
        self.area = []
        self.set_contour(contours)

    def set_find(self, find):
        self.checked = find

    def set_die(self):
        self.die = True

    def set_contour(self, contours):
        x1 = np.max(np.asarray(contours[:,0, 0]))
        x2 = np.min(np.asarray(contours[:,0, 0]))

        y1 = np.max(np.asarray(contours[:,0, 1]))
        y2 = np.min(np.asarray(contours[:,0, 1]))
        self.center.append(((x1+x2)/2.,(y1+y2)/2.))
        self.base = cv2.drawContours(np.zeros((64,64), np.uint8), [contours], 0, 255, -1)
        self.area.append(np.sum(self.base//255))

    def update_info(self, contour):
        self.set_contour(contour)


class TrackingInfoMaker:

    def __init__(self):
        self.random_id = 0
        self.contours = {}

    def create_new_id(self):
        return len(self.contours)+1

    def create_info(self, contour):
        new_id = self.create_new_id()
        self.contours[new_id] = TrackingInfo(contour, new_id)
        return self.contours[new_id]

    def update_tracking_info(self, contours):
        contour_count = len(contours)
        is_used = [False for _ in range(contour_count)]
        base = np.zeros((64,64), np.uint8)
        contour_base = [cv2.drawContours(base.copy(), contours, i, 255, -1) for i in range(contour_count)]
        for ct_key in self.contours:
            if self.contours[ct_key].die:
                continue
            else:
                self.contours[ct_key].set_find(False)
                max_duplicate = 0
                max_idx = -1
                for idx in range(contour_count):
                    if is_used[idx]:
                        continue

                    duplicate = np.sum(contour_base[idx]//255 * self.contours[ct_key].base//255)

                    # if duplicate > max_duplicate and duplicate > self.contours[ct_key].area[-1]//2:
                    if duplicate > max_duplicate:
                        max_duplicate = duplicate
                        max_idx = idx

                if max_idx != -1:
                    is_used[max_idx] = True
                    self.contours[ct_key].update_info(contours[max_idx])
                    self.contours[ct_key].set_find(True)
                else:
                    self.contours[ct_key].set_die()

        for idx in range(contour_count):
            if not is_used[idx]:
                self.create_info(contours[idx])


def predict_to_image(predicted):
    np_image = predicted.detach().cpu().numpy()
    np_image = np.squeeze(np_image)
    np_image = np.expand_dims(np_image, -1)
    np_image = np.where(np_image > 0.5, 255, 0)
    np_image = np.array(np_image, dtype=np.uint8)
    print(np_image.shape)
    return np_image


def get_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for ct in contours:
        print(ct.shape)
    return contours


def inference(model_path, video_path):
    def convert_image_for_input(image):
        image_shape = image.shape
        print(image_shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = image/255.
        image = torch.from_numpy(image).type(torch.FloatTensor).cuda()

        return image
    model = torch.jit.load(model_path)

    if torch.cuda.is_available():
        model.cuda()
    maker = TrackingInfoMaker()
    vc = cv2.VideoCapture(video_path)
    ret, image = vc.read()
    while ret:
        # image = image/255
        image = convert_image_for_input(image)
        predict = model(image)
        print(torch.max(predict))
        predict = predict_to_image(predict)
        contours = get_contour(predict)
        cv2.imshow('result', predict)

        maker.update_tracking_info(contours)
        input_image = image.detach().cpu().numpy()
        input_image = np.squeeze(input_image)
        input_image = np.transpose(input_image, (1,2,0))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        input_image = np.array(input_image*255, dtype=np.uint8)
        draw_image = input_image.copy()
        print(input_image.shape)
        for ct_key in maker.contours:
            if maker.contours[ct_key].die:
                continue
            else:
                center = maker.contours[ct_key].center[-1]
                cv2.putText(img=draw_image, text='{}'.format(ct_key),
                            org=(int(center[0]*4), int(center[1]*4)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255,255,255),
                            lineType=2)
                resized_base = cv2.resize(maker.contours[ct_key].base, (256,256))
                edge = cv2.Canny(resized_base,128, 255)
                edge = np.expand_dims(edge, axis=-1)
                edge = np.ones((256,256,3), dtype=np.uint8)*edge
                draw_image = draw_image & (255-edge)
                draw_image = draw_image | edge

        cv2.imshow('draw', draw_image)
        cv2.waitKey(10)
        ret, image = vc.read()


if __name__ == "__main__":
    model_path = 'D:\\dataset\\Droplet\\best_model.zip'
    video_path = 'D:\\dataset\\Droplet\\a.avi'
    inference(model_path, video_path)
