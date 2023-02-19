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
    image = scipy.ndimage.interpolation.zoom(image,
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


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):

    # 不是二进制，而是1和2。
    # 0 被视为背景，这是我们不需要的
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # 选择最角落的像素来确定哪个标签是空气
    # 优化:从患者周围选取多个背景标签
    background_label = labels[0, 0, 0]

    # 填充人体周边空气
    binary_image[background_label == labels] = 2

    # 填充肺部结构的方法 (比形态闭合之类的方法要好)
    if fill_lung_structures:

        # 对于每个切片，我们确定最大的固体结构
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # 这个切片包含一些肺部组织
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # 使图像实际二进制
    binary_image = 1-binary_image  # 反过来，肺现在是 1

    # 去除阀体内的其他空气
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # 如果有气穴
        binary_image[labels != l_max] = 0

    return binary_image


dicom_folder = './data/'
patients = os.listdir(dicom_folder)

sample_patient = load_dicom(dicom_folder + patients[0])
sample_patient_pixels = get_hu(sample_patient)

pix_resampled, spacing = resample(
    sample_patient_pixels, sample_patient, [1, 1, 1])

segmented_lungs = segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, True)


start_sum = time.time()  # 记录3D渲染开始时间

# plot_3d(segmented_lungs, 0)
# plot_3d(segmented_lungs_fill, 0) # 在肺内包含结构（因为结节是固体），在肺部不只有空气。
plot_3d(segmented_lungs_fill - segmented_lungs, 0)

end_sum = time.time()  # 记录3D渲染结束时间
print('渲染总耗时：', end_sum - start_sum, '秒')
