import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    # 定义卷积层，包括卷积操作、批归一化和激活函数。

    # 参数：
    # - batchNorm: 是否使用批归一化（True/False）
    # - in_planes: 输入通道数
    # - out_planes: 输出通道数
    # - kernel_size: 卷积核大小，默认为3
    # - stride: 步长，默认为1

    # 返回：
    # - nn.Sequential对象，包含卷积层、批归一化和激活函数
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def upconv(in_planes, out_planes):
    # 定义上采样卷积层，包括转置卷积操作和ReLU激活函数。

    # 参数：
    # - in_planes: 输入通道数
    # - out_planes: 输出通道数

    # 返回：
    # - nn.Sequential对象，包含转置卷积层和ReLU激活函数
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

def predict_flow(in_planes):
    # 定义预测光流的卷积层。

    # 参数：
    # - in_planes: 输入通道数

    # 返回：
    # - nn.Conv2d对象，用于预测光流
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

class FlowNet(nn.Module):

    def __init__(self, batchNorm=False):
        super(FlowNet, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def project_to_cylindrical(self, point_cloud, height, width):
        # 将点云投影到柱面坐标的逻辑
        # point_cloud: 输入的点云数据，形状为 (batch_size, num_points, 3)
        # height: 柱面图像的高度
        # width: 柱面图像的宽度

        # 假设你的投影逻辑是简单地将点云的 x 和 y 坐标映射到柱面图像的 x 和 y 坐标
        x = point_cloud[:, :, 0]  # 提取点云的 x 坐标
        y = point_cloud[:, :, 1]  # 提取点云的 y 坐标

        # 将 x 和 y 坐标映射到柱面图像的 x 和 y 坐标范围内
        x_mapped = (x + 1) * (width - 1) / 2  # 映射到 [0, width-1] 范围内
        y_mapped = (y + 1) * (height - 1) / 2  # 映射到 [0, height-1] 范围内

        # 创建一个空的柱面图像张量
        cylindrical_image = torch.zeros((point_cloud.shape[0], height, width))

        # 将点云投影到柱面图像中
        cylindrical_image.scatter_(2, x_mapped.unsqueeze(2).long(), y_mapped.unsqueeze(2).long(), 1)

        return cylindrical_image
    def extract_point_cloud_features(self, cylindrical_image):
        # 使用FlowNet提取点云特征的逻辑
        # cylindrical_image: 柱面图像，形状为 (batch_size, height, width)

        # 假设你的特征提取逻辑是简单地将柱面图像展平为一维向量
        features = cylindrical_image.view(cylindrical_image.shape[0], -1)

        return features

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, target_image, ref_img):

        s = target_image.size()

        input = torch.cat([target_image, ref_img], 1)

        out_conv1 = self.conv1(input)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        features = out_conv6.view(1, s[0], -1)
        
        # 将点云数据投影到柱面坐标中
        cylindrical_image = self.project_to_cylindrical(point_cloud, height, width)

        # 使用FlowNet提取点云特征
        point_cloud_features = self.extract_point_cloud_features(cylindrical_image)

        return features, point_cloud_features

class RecFeat(nn.Module):
    # LSTM
    def __init__(self, x_dim, h_dim, batch_size, n_layers):

        super(RecFeat, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.output_dim = h_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=h_dim, num_layers=n_layers, dropout=0.2)
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers, self.batch_size, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers, self.batch_size, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, self.batch_size, self.h_dim)),
                    Variable(torch.zeros(self.n_layers, self.batch_size, self.h_dim)))

    def init_test_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, 1, self.h_dim)),
                    Variable(torch.zeros(self.n_layers, 1, self.h_dim)))

    def forward(self, x):

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.view(-1, self.h_dim)

        return lstm_out

class RecImu(nn.Module):

    def __init__(self, x_dim, h_dim, batch_size, n_layers, output_dim):

        super(RecImu, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=h_dim, num_layers=n_layers, dropout=0.2, bidirectional=True)
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden()
        self.output_dim = output_dim

    def init_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)),
                    Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)))

    def init_test_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)),
                    Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)))

    def forward(self, x):

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #lstm_out = self.dropout(lstm_out)
        result = lstm_out[-1].view(-1, self.h_dim * 2)

        result = result.view(1, -1, self.output_dim)

        return result

class Fc_Flownet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Fc_Flownet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim)
        )

        self.dropout = nn.Dropout(0.2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat):

        feat = feat.view(-1, self.input_dim)

        feat_new = self.fc(feat).view(1, -1, self.output_dim)

        #feat_new = self.dropout(feat_new)

        return feat_new

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, latent_dim):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim, 2)

class Hard_Mask(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Hard_Mask, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim * 2),
            nn.ReLU()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat, temp):

        feat = feat.view(-1, self.input_dim)

        feat_new = self.fc(feat).view(-1, self.output_dim, 2)

        z = gumbel_softmax(feat_new, temp, self.output_dim)

        z1 = z[:, :, 1].view(1, -1, self.output_dim)

        return z1

class Soft_Mask(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Soft_Mask, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat):

        feat = feat.view(-1, self.input_dim)

        feat_new = self.fc(feat).view(1, -1, self.output_dim)

        #feat_new = self.dropout(feat_new)

        return feat_new

class PoseRegressor(nn.Module):
    # 姿态回归器模型，用于预测姿态（位移和旋转）

    # 参数：
    # - feature_dim: 输入特征的维度

    # 成员变量：
    # - feature_dim: 输入特征的维度
    # - tr_pred: 位移预测网络
    # - ro_pred: 旋转预测网络
    # - dropout: Dropout层

    # 方法：
    # - init_weights: 初始化模型权重
    # - forward: 前向传播计算姿态预测结果
    def __init__(self, feature_dim):
        super(PoseRegressor, self).__init__()

        self.feature_dim = feature_dim
        self.tr_pred = nn.Sequential(
            nn.Linear(self.feature_dim, 3)
        )
        self.ro_pred = nn.Sequential(
            nn.Linear(self.feature_dim, 3)
        )

        self.dropout = nn.Dropout(0.2)

    def init_weights(self):
        # 初始化模型权重，使用Xavier初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat):
        # 前向传播计算姿态预测结果

        # 参数：
        # - feat: 输入特征

        # 返回：
        # - pose: 姿态预测结果
        feat = feat.view(-1, self.feature_dim)

        tr = self.tr_pred(feat)
        ro = self.ro_pred(feat)

        pose = torch.cat((tr, ro), 1)

        pose = pose.view(pose.size(0), 1, 6)

        return pose
