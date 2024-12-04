import numpy as np
import math
from pathlib import Path
import struct
from io import BufferedReader
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 导入MNIST数据集的路径
train_img_path = './mnist/train-images.idx3-ubyte'
train_label_path = './mnist/train-labels.idx1-ubyte'
test_img_path = './mnist/t10k-images.idx3-ubyte'
test_label_path = './mnist/t10k-labels.idx1-ubyte'

# 数据集参数
train_num = 5000
valid_num = 1000
test_num = 1000

# 数据集分类函数
def file_trans_np(file_path, sel):
    with open(file_path, 'rb') as f:
        if sel == 'img':
            struct.unpack('>4i', f.read(16))  # 跳过头部
            rs = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255.0
        if sel == 'lab':
            struct.unpack('>2i', f.read(8))  # 跳过头部
            rs = np.fromfile(f, dtype=np.uint8)
    return rs

# HOG特征提取函数
def compute_hog_features(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9, visualize=False, feature_vector=True):
    """
    计算单张图像的HOG特征。

    参数:
    - image: 一维numpy数组，长度为784（28x28像素）。
    - pixels_per_cell: 每个cell的像素大小。
    - cells_per_block: 每个block包含的cell数量。
    - orientations: 方向梯度的数量。
    - visualize: 是否返回HOG图像。
    - feature_vector: 是否返回特征向量。

    返回:
    - features: HOG特征向量。
    - hog_image (可选): HOG图像（如果visualize=True）。
    """
    image = image.reshape((28, 28))  # 重塑为二维
    if visualize:
        features, hog_image = hog(image,
                                  orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  block_norm='L2-Hys',
                                  visualize=visualize,
                                  feature_vector=feature_vector)
        return features, hog_image
    else:
        features = hog(image,
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      block_norm='L2-Hys',
                      visualize=visualize,
                      feature_vector=feature_vector)
        return features

# 批量提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for img in tqdm(images, desc="Extracting HOG Features"):
        features = compute_hog_features(img)
        hog_features.append(features)
    return np.array(hog_features)

# 加载原始数据
train_img_raw = file_trans_np(train_img_path, 'img')[:train_num]
train_lab = file_trans_np(train_label_path, 'lab')[:train_num]
valid_img_raw = file_trans_np(train_img_path, 'img')[train_num:train_num + valid_num]
valid_lab = file_trans_np(train_label_path, 'lab')[train_num:train_num + valid_num]
test_img_raw = file_trans_np(test_img_path, 'img')[:test_num]
test_lab = file_trans_np(test_label_path, 'lab')[:test_num]

# 提取HOG特征
train_img = extract_hog_features(train_img_raw)
valid_img = extract_hog_features(valid_img_raw)
test_img = extract_hog_features(test_img_raw)

print(f"HOG Feature Dimension: {train_img.shape[1]}")

# 激活函数
def bypass(x):
    return x

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()

# 激活函数的导数
def d_bypass(data):
    return 1

def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm, sm)

def d_tanh(data):
    return 1 / (np.cosh(data))**2

# 模型结构参数
differential = {softmax: d_softmax, tanh: d_tanh, bypass: d_bypass}
d_type = {bypass: 'times', softmax: 'dot', tanh: 'times'}

# 神经网络结构
hog_feature_length = train_img.shape[1]
dimensions = [hog_feature_length, 100, 10]
activation = [bypass, tanh, softmax]

# 权重初始化分布
distribution = [
    {},  # 输入层没有参数
    {
        'b': [0, 0],
        'w': [-math.sqrt(6 / (dimensions[0] + dimensions[1])), math.sqrt(6 / (dimensions[0] + dimensions[1]))]
    },
    {
        'b': [0, 0],
        'w': [-math.sqrt(6 / (dimensions[1] + dimensions[2])), math.sqrt(6 / (dimensions[1] + dimensions[2]))]
    }
]

# 初始化偏置
def init_parameter_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer]) * (dist[1] - dist[0]) + dist[0]

# 初始化权重
def init_parameter_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer - 1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]

# 初始化参数
def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_parameter['b'] = init_parameter_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameter_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter

# 预测函数
def predict(img, parameters):
    l_in = img
    l_out = activation[0](l_in)  # Bypass
    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out, parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
    return l_out

# One-hot编码
onehot = np.identity(dimensions[-1])

# 平方损失函数
def sqr_loss(img, lab, parameters):
    y_pred = predict(img, parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff, diff)

# 计算梯度
def grad_parameters(img, lab, parameters):
    l_in_list = [img]
    l_out_list = [activation[0](l_in_list[0])]

    for layer in range(1, len(dimensions)):
        l_in = np.dot(l_out_list[layer - 1], parameters[layer]['w']) + parameters[layer]['b']
        l_out = activation[layer](l_in)
        l_in_list.append(l_in)
        l_out_list.append(l_out)

    d_layer = -2 * (onehot[lab] - l_out_list[-1])
    grad_result = [None] * len(dimensions)

    # 反向传播
    for layer in range(len(dimensions) - 1, 0, -1):
        if d_type[activation[layer]] == 'times':
            d_layer = differential[activation[layer]](l_in_list[layer]) * d_layer
        if d_type[activation[layer]] == 'dot':
            d_layer = np.dot(differential[activation[layer]](l_in_list[layer]), d_layer)
        grad_result[layer] = {}
        grad_result[layer]['b'] = d_layer
        grad_result[layer]['w'] = np.outer(l_out_list[layer - 1], d_layer)
        d_layer = np.dot(parameters[layer]['w'], d_layer)

    return grad_result

# 梯度测试（可选）
def grad_test(parameters, layer, pname, h=1e-5):
    grad_list = []
    for i in tqdm(range(len(parameters[layer][pname]))):
        for j in range(len(parameters[layer][pname][0])):
            img_i = np.random.randint(train_num)
            test_parameters = init_parameters()
            derivative = grad_parameters(train_img[img_i], train_lab[img_i], test_parameters)[layer][pname]
            value1 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
            test_parameters[layer][pname][i][j] += h
            value2 = sqr_loss(train_img[img_i], train_lab[img_i], test_parameters)
            grad_list.append(derivative[i][j] - (value2 - value1) / h)
    print(f"Max gradient difference: {np.abs(grad_list).max()}")

# 损失函数
def loss(parameters, types):
    loss_accu = 0
    if types == "train":
        for img_i in range(train_num):
            loss_accu += sqr_loss(train_img[img_i], train_lab[img_i], parameters)
        loss_accu = loss_accu / (train_num / 10000)
    if types == "valid":
        for img_i in range(valid_num):
            loss_accu += sqr_loss(valid_img[img_i], valid_lab[img_i], parameters)
        loss_accu = loss_accu / (valid_num / 10000)
    return loss_accu

# 计算准确率
def accuracy(parameters, types, top_n=1):
    correct = 0
    dataset_size = len(train_img) if types == "train" else len(valid_img) if types == "valid" else len(test_img)
    for img_i in range(dataset_size):
        if types == "train":
            img = train_img[img_i]
            lab = train_lab[img_i]
        elif types == "valid":
            img = valid_img[img_i]
            lab = valid_lab[img_i]
        elif types == "test":
            img = test_img[img_i]
            lab = test_lab[img_i]

        pred = predict(img, parameters)

        if top_n == 1:
            if pred.argmax() == lab:
                correct += 1
        elif top_n == 5:
            top_5_preds = pred.argsort()[-5:][::-1]
            if lab in top_5_preds:
                correct += 1

    return correct / dataset_size

# 批量大小
batch_size = 15

# 梯度累加
def grad_count(grad1, grad2, types):
    for layer in range(1, len(grad1)):
        for pname in grad1[layer].keys():
            if types == "add":
                grad1[layer][pname] += grad2[layer][pname]
            if types == "divide":
                grad1[layer][pname] /= grad2
    return grad1

# 训练一个批次
def train_batch(current_batch, parameters):
    grad_accu = grad_parameters(train_img[current_batch * batch_size + 0], train_lab[current_batch * batch_size + 0], parameters)
    for img_i in range(1, batch_size):
        grad_tmp = grad_parameters(train_img[current_batch * batch_size + img_i], train_lab[current_batch * batch_size + img_i], parameters)
        grad_count(grad_accu, grad_tmp, "add")
    grad_count(grad_accu, batch_size, "divide")
    return grad_accu

# 更新参数
def combine_parameters(parameters, grad, learn_rate):
    parameters_tmp = copy.deepcopy(parameters)
    for layer in range(1, len(parameters_tmp)):
        for pname in parameters_tmp[layer].keys():
            parameters_tmp[layer][pname] -= learn_rate * grad[layer][pname]
    return parameters_tmp

# 学习率可视化
def learn_rate_show(parameters, lower=0, upper=1, step=0.1, batch=np.random.randint(train_num // batch_size)):
    grad_lr = train_batch(batch, parameters)
    lr_list = []
    for learn_rate in tqdm(np.linspace(lower, upper, num=int((upper - lower) / step + 1)), desc="Testing Learning Rates"):
        parameters_tmp = combine_parameters(parameters, grad_lr, learn_rate)
        loss_tmp = loss(parameters_tmp, "train")
        lr_list.append([learn_rate, loss_tmp])
    lr_list = np.array(lr_list)
    plt.plot(lr_list[:, 0], lr_list[:, 1], color="deepskyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs Loss")
    plt.show()

# 统计数据可视化
def statistics_show(img_list: list, color_list: list, title, lower=0):
    plt.title(title)
    for i in range(len(img_list)):
        plt.plot(img_list[i][lower:], color=color_list[i], label=f"{title} {i}" if len(img_list) > 1 else title)
    plt.legend()
    plt.show()

# 更新统计数据
def statistics_count(parameters, statistics_list):
    for key in statistics_list:
        statistics_list[key]["loss"].append(loss(parameters, key))
        statistics_list[key]["accu"].append(accuracy(parameters, key))
    return statistics_list

# 训练函数
def train(parameters, learn_rate, epoch_num, statistics=False):
    statistics_list = {"train": {"loss": [], "accu": []}, "valid": {"loss": [], "accu": []}}

    for epoch in tqdm(range(epoch_num), desc="Epochs"):
        for i in tqdm(range(train_num // batch_size), desc="Batches", leave=False):
            grad_tmp = train_batch(i, parameters)
            parameters = combine_parameters(parameters, grad_tmp, learn_rate)

        if statistics:
            statistics_list = statistics_count(parameters, statistics_list)

    if statistics:
        statistics_show([statistics_list["train"]["loss"], statistics_list["valid"]["loss"]], ["black", "red"], "Loss")
        statistics_show([statistics_list["train"]["accu"], statistics_list["valid"]["accu"]], ["black", "red"], "Accuracy")

    return parameters

# 预测结果可视化
def predict_show(parameters, count, img_type=test_img, img_num=test_num):
    test_list = np.random.choice(range(0, img_num), count, replace=False)
    for i in test_list:
        pred_label = predict(img_type[i], parameters).argmax()
        print(f"Prediction: {pred_label}")
        show_img(i, 'test', pred_label)

# 显示图像函数
def show_img(index, types, predict_value=None):
    if types == 'train':
        img = train_img_raw[index].reshape(28, 28)
        lab = train_lab[index]
    if types == 'valid':
        img = valid_img_raw[index].reshape(28, 28)
        lab = valid_lab[index]
    if types == 'test':
        img = test_img_raw[index].reshape(28, 28)
        lab = test_lab[index]
    plt.title(f"Label: {lab}  Prediction: {predict_value}")
    plt.imshow(img, cmap="gray")
    plt.show()

# 混淆矩阵可视化
def confusion_matrix_show(parameters, img_type=test_img, img_num=test_num):
    true_labels = []
    predicted_labels = []

    for img_i in tqdm(range(img_num), desc="Generating Confusion Matrix Data"):
        true_labels.append(test_lab[img_i])
        predicted_labels.append(predict(img_type[img_i], parameters).argmax())  # 修正这里的 'i' 为 'img_i'

    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# 初始化模型参数
parameters = init_parameters()

# 设置学习率和训练轮次
learn_rate = 0.15
epoch_num = 15

# 训练模型
parameters = train(parameters, learn_rate, epoch_num, True)

# 评估验证集准确率
print(f"Validation Accuracy: {accuracy(parameters, 'valid') * 100:.2f}%")

# 可视化部分预测结果
predict_show(parameters, 5)

# 测试集准确率
print(f"Test Accuracy (Top-1): {accuracy(parameters, 'test', 1) * 100:.2f}%")
print(f"Test Accuracy (Top-5): {accuracy(parameters, 'test', 5) * 100:.2f}%")

# 保存模型参数（可选）
# np.save("parameters.npy", parameters)

# 显示混淆矩阵
confusion_matrix_show(parameters)
