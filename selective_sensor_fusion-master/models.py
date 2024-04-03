import torch
import math
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

    def __init__(self, batchNorm=False,num_point_features=3):
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

        # 添加新的卷积层或全连接层来处理点云特征
        self.conv_point_cloud = conv(self.batchNorm, num_point_features, num_point_features, kernel_size=3, stride=1)

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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, target_image, ref_img, point_cloud):
        s = target_image.size()
        input = torch.cat([target_image, ref_img], 1)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # 提取点云特征
        point_cloud_features = self.conv_point_cloud(point_cloud)

        # 融合图像特征和点云特征
        fused_features = torch.cat([out_conv6, point_cloud_features], dim=1)

        features = out_conv6.view(1, s[0], -1)

        # 将点云数据投影到柱面坐标中
        height, width = self.calculate_height_width(point_cloud)  # 根据点云数据计算图像的高度和宽度
        cylindrical_image = self.project_to_cylindrical(point_cloud, height, width)

        # 使用FlowNet提取点云特征
        point_cloud_features = self.extract_point_cloud_features(cylindrical_image)

        return fused_features, point_cloud_features

class PointCloud(nn.Module):
    def __init__(self, num_points):
        self.num_points = num_points

    def project_to_cylindrical(self, point_cloud, height, width):
        x = point_cloud[:, :, 0]  # 提取点云的 x 坐标
        y = point_cloud[:, :, 1]  # 提取点云的 y 坐标
        z = point_cloud[:, :, 2]  # 提取点云的 z 坐标
        alpha = torch.atan2(y, x)  # 计算 alpha
        beta = torch.asin(z / torch.sqrt(x**2 + y**2 + z**2))  # 计算 beta

        # 将 alpha 和 beta 映射到柱面图像的坐标范围内
        alpha_mapped = (alpha / (2 * math.pi) + 0.5) * width  # 映射到 [0, width] 范围内
        beta_mapped = (beta + 0.5) * height  # 映射到 [0, height] 范围内

        # 创建一个空的柱面图像张量
        cylindrical_image = torch.zeros((point_cloud.shape[0], height, width))

        # 将点云投影到柱面图像中
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                alpha_idx = int(alpha_mapped[i, j])
                beta_idx = int(beta_mapped[i, j])
                if alpha_idx >= 0 and alpha_idx < width and beta_idx >= 0 and beta_idx < height:
                    if cylindrical_image[i, beta_idx, alpha_idx] == 0 or cylindrical_image[i, beta_idx, alpha_idx] > torch.sqrt(x[i, j]**2 + y[i, j]**2 + z[i, j]**2):
                        cylindrical_image[i, beta_idx, alpha_idx] = torch.sqrt(x[i, j]**2 + y[i, j]**2 + z[i, j]**2)
        return cylindrical_image

    def calculate_height_width(self, point_cloud):
        min_x = torch.min(point_cloud[:, :, 0])
        max_x = torch.max(point_cloud[:, :, 0])
        min_y = torch.min(point_cloud[:, :, 1])
        max_y = torch.max(point_cloud[:, :, 1])

        height = max_y - min_y
        width = max_x - min_x

        return height, width

    def extract_point_cloud_features(self, cylindrical_image):
        # 将柱面图像展开为一维向量
        features = cylindrical_image.view(cylindrical_image.shape[0], -1)

        return features
 
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

    def forward(self, x, point_cloud):
        # 将点云数据与输入特征进行融合
        x = torch.cat((x, point_cloud), dim=2)
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

    def forward(self, feat, point_cloud):
        # 前向传播计算姿态预测结果
        feat = feat.view(-1, self.feature_dim)
        point_cloud = point_cloud.view(-1, self.point_cloud_dim)

        feat_with_pc = torch.cat((feat, point_cloud), dim=1)

        tr = self.tr_pred(feat_with_pc)
        ro = self.ro_pred(feat_with_pc)

        pose = torch.cat((tr, ro), 1)
        pose = pose.view(pose.size(0), 1, 6)

        return pose
