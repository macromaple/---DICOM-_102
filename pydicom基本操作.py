# 加载基础数据科学库
import os  # 系统交互库
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 机器学习图像库
from skimage import measure, morphology  # 机器学习图像库
import scipy.ndimage  # 多维图像处理
import pydicom  # 加载 pydicom
import numpy as np  # 线性代数计算库
import pandas as pd  # 数据处理库，如读取 csv 文件
import matplotlib.pyplot as plt  # 数据可视化库

plt.rcParams['font.family'] = 'sans-serif'    # 用来正常显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 设置正常显示符号

# 基本参数定义
dicom_folder = './data/'  # 数据集存储路径
PathDicom = './data/sample_images'     # 批量读取单一病例目录
patients = os.listdir(dicom_folder)    # 列出数据集目录下子目录
patients.sort()                        # 列出数据集目录



# DICOM 文件路径
ds = pydicom.read_file(
    dicom_folder + '/sample_images/0acb5ac56995154ffe0344fe319c876b.dcm')



# 每一个 DICOM 标签都是由两个十六进制数的组合来确定的，分别为 `Group` 和 `Element`。如 `(0010,0010)` 这个标签表示的是 `Patient's Name`，它存储着这张 DICOM 图像的患者姓名。`pydicom.read_file` 函数已经为我们完成了解析，输出所有 DICOM 元数据标签名称。
print(ds.dir())



# 输出包含 `Patient` 的 DICOM 元数据标签，提取患者信息。
print(ds.dir('Patient'))



# 输出 DICOM 元数据标签相应的属性值。
print(ds.ImageOrientationPatient, ds.ImagePositionPatient,
      ds.PatientBirthDate, ds.PatientID, ds.PatientName)



# 输出完整的数据元素，包括 DICOM 标签编码值（Group, Element）, VR, Value。其中，VR 是 DICOM 标准中用来描述数据类型的，总共有 27 个值。
print(ds.data_element('PatientID'))



# `VR=LO` 长字符串，这里实际上是对患者 ID 的哈希化隐藏。
print(ds.data_element('PatientID').VR, ds.data_element('PatientID').value)



# 输出单层 DICOM 图片与形状。
pixel_bytes = ds.PixelData  # 原始二进制文件
pix = ds.pixel_array       # 像素值矩阵
print(pix.shape)           # 输出矩阵维度

# cmap 表示 colormap,
# 可以是设置成不同值获得不同显示效果,输出 dicom 图片
plt.imshow(pix, cmap=plt.cm.gray)
plt.show()



# 转换为完整切片组合的三维数组。
# 用 lstFilesDCM 作为存放 DICOM files 的列表
lstFilesDCM = []

# 将所有 DICOM 文件读入
for diName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # 判断文件是否为 dicom 文件
            # print(filename)
            lstFilesDCM.append(os.path.join(diName, filename))  # 加入到列表中

# 将第一张图片作为参考图
RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张 DICOM 图片

# 建立三维数组,分别记录长、宽、层数(也就是该患者 DICOM 样本个数)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
print(ConstPixelDims)
