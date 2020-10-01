import torch.nn as nn

from torch_module.layers import BottleNeckBlock, Conv2D, UpConv2D
from torch_module.utils import get_param_count
import torch


class HRNet(nn.Module):
    def __init__(self, feature, depth, input_ch, output_ch, out_act, act):
        super(HRNet, self).__init__()
        self.feature = feature
        self.input_ch = input_ch
        self.depth = depth
        self.out_act = out_act

        self.border_1 = self.depth // 3
        self.border_2 = self.depth // 3 * 2

        if type(output_ch) == int:
            self.output_ch = [output_ch]
        else:
            self.output_ch = output_ch
        self.out_len = len(self.output_ch)

        self.__build__(act)

    def __build__(self, act):
        self.conv_7 = Conv2D(self.input_ch, self.feature, 7, 2, 1, activation=act, batch=False)

        self.bottle_1 = BottleNeckBlock(self.feature, attention=True, activation=act)

        self.first_stair = nn.ModuleList()
        self.second_stair = nn.ModuleList()
        self.third_stair = nn.ModuleList()
        for j in range(self.depth):
            self.first_stair.append(BottleNeckBlock(input_feature=self.feature, attention=True))
            if self.border_1 == j:
                self.second_stride_1_2 = Conv2D(self.feature, self.feature*2, 3, 2, 1, activation=act)

            if self.border_1 < j:
                self.second_stair.append(BottleNeckBlock(input_feature=self.feature*2, attention=True, activation=act))

            if self.border_2 == j:
                self.third_stride_1_2 = Conv2D(self.feature, self.feature*2, 3, 2, 1, activation=act)
                self.third_stride_1_3 = Conv2D(self.feature, self.feature*3, 3, 4, 1, activation=act)
                self.third_stride_2_3 = Conv2D(self.feature*2, self.feature*3, 3, 2, 1, activation=act)
                self.up_stride_2_1_1 = UpConv2D(2, self.feature*2, self.feature, 1, 1, 0, activation=act)

            if self.border_2 < j:
                self.third_stair.append(BottleNeckBlock(input_feature=self.feature*3, attention=True, activation=act))

        self.hr_last = BottleNeckBlock(self.feature, attention=True, activation=act)

        self.last_front = nn.ModuleList([BottleNeckBlock(self.feature, attention=True, activation=act) for _ in range(self.out_len)])
        self.last = nn.ModuleList([Conv2D(self.feature, self.output_ch[i], 1, stride=1, padding=0, activation=self.out_act[i]) for i in range(self.out_len)])
        # self.last_1 = BottleNeckBlock(self.feature, attention=True, activation=act)
        # self.last = Conv2D(self.feature, self.output_ch, 1, stride=1, padding=0, activation='sigmoid')

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.up_stride_2_1_2 = UpConv2D(2, self.feature * 2, self.feature, 1, 1, 0, activation=act)
        self.up_stride_3_2_1 = UpConv2D(2, self.feature * 3, self.feature*2, 1, 1, 0, activation=act)
        self.up_stride_2_1_3 = UpConv2D(2, self.feature * 2, self.feature, 1, 1, 0, activation=act)

        self.batch = nn.BatchNorm1d(256)

        self.stem = nn.Sequential(
            self.conv_7,
            self.bottle_1,
            self.max_pool
        )

    def forward(self, x):
        x = self.stem(x)
        first_output = x
        second_output = None
        third_output = None
        for i in range(self.depth):
            first_output = self.first_stair[i](first_output)
            if self.border_1 == i:
                second_output = self.second_stride_1_2(first_output)

            elif self.border_1 < i:
                second_output = self.second_stair[i-self.border_1-1](second_output)

            if self.border_2 == i:
                first_up = self.up_stride_2_1_1(second_output)

                second_output_1 = self.third_stride_1_2(first_output)

                third_output_1 = self.third_stride_1_3(first_output)
                third_output_2 = self.third_stride_2_3(second_output)

                first_output = first_output + first_up
                second_output = second_output + second_output_1
                third_output = third_output_1 + third_output_2

            elif self.border_2 < i:
                third_output = self.third_stair[i-self.border_2-1](third_output)

        last_1 = first_output
        last_2 = self.up_stride_2_1_2(second_output)
        last_3 = self.up_stride_2_1_3(self.up_stride_3_2_1(third_output))
        last_input = last_1 + last_2 + last_3
        last_output = self.hr_last(last_input)

        predict_output = []
        for i in range(self.out_len):
            out = self.last_front[i](last_output)
            out = self.last[i](out)
            predict_output.append(out)
        # predict_output = self.last_1(last_output)
        # predict_output = self.last(predict_output)

        return predict_output


if __name__ == "__main__":
    ip = torch.rand((2,6,256,256)).cuda()
    net = HRNet(feature=256, depth=7, input_ch=6, output_ch=[15,28], out_act=['sigmoid', 'sigmoid'], act='selu').cuda()
    print(get_param_count(net))
    output = net(ip)
    print(output[0].shape)
    # print(output[1].shape, output[1])

