import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
import time


def load_dicom(path):
    # 在给定的文件夹路径中加载 DICOM 扫描影像
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    # 对患者所有的 DICOM 影像进行排序
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 计算切片厚度
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)

    # 添加缺失的元数据: 切片厚度
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_hu(slices):
    image = np.stack([s.pixel_array for s in slices])

    # 转换为 int16，int16 是可接受的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)

    # 设置边界外的元素为 0
    # 截距通常为 -1024，因此空气约为 0
    image[image == -2000] = 0

    # 转换为 HU 单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

    # 乘以重新缩放斜率
        if slope != 1:
            image[slice_number] = slope * \
                image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        # 添加截距
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # 确定当前像素间距
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing),
                       dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # 使用 scipy.ndimage.interpolation.zoom 进行数组缩放
    # Please use `zoom` from the `scipy.ndimage` namespace, the `scipy.ndimage.interpolation` namespace is deprecated.
    image = scipy.ndimage.zoom(image,
                               real_resize_factor,
                               mode='nearest')  # 插值模式

    return image, new_spacing


def plot_3d(image, threshold=-300):

    # 垂直放置扫描，
    # 因此，患者的头部会在上方，面对着摄像机
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)  # 有改动

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 使用`verts[faces]` 生成三角形集合
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


dicom_folder = 'F:/bysj/---DICOM-_102/data/'
patients = os.listdir(dicom_folder)
print(patients)

# 加载第一个患者的扫描片目录，由于实验数据集内只有一个患者，选择 0
sample_patient = load_dicom(dicom_folder + patients[0])
print("load_dicom OK")
sample_patient_pixels = get_hu(sample_patient)
print("get_hu OK")
# plt.hist(sample_patient_pixels.flatten(), bins=80, color='c')
# plt.xlabel("HU 单位 (Hounsfield Units)")
# plt.ylabel("频率")
# plt.show()

# 其中，挑选一个中间位置的切片进行显示
# plt.imshow(sample_patient_pixels[100], cmap=plt.cm.gray)
# plt.show()


pix_resampled, spacing = resample(
    sample_patient_pixels, sample_patient, [1, 1, 1])
print("resample OK")
# print("重采样前的形状\t", sample_patient_pixels.shape)
# print("重采样后的形状\t", pix_resampled.shape)


start_sum = time.time()  # 记录3D渲染开始时间

# 定义 HU 阈值，仅将骨骼结构渲染为三维图像
plot_3d(pix_resampled, 400)
print("plot_3d OK")

end_sum = time.time()  # 记录3D渲染结束时间
print('渲染总耗时：', end_sum - start_sum, '秒')
