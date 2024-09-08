from rknnlite.api import RKNNLite
import cv2
import numpy as np


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    rknn = RKNNLite()

    # 使用load_rknn接口直接加载RKNN模型
    rknn.load_rknn(path="./resnet18.rknn")

    # 调用init_runtime接口初始化运行时环境
    rknn.init_runtime(
        core_mask=0,  # 表示NPU的调度模式
    )

    # 使用opencv获取推理的图片数据
    img = cv2.imread(filename="./space_shuttle_224.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #调用inference接口进行推理测试
    outputs = rknn.inference(
        inputs=[img],
        data_format=None
    )

    show_outputs(softmax(np.array(outputs[0][0])))

    rknn.release()