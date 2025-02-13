import numpy as np
import matplotlib.pyplot as plt
import pylab

# 加载NumPy文件
# numpy_file_path_image = 'E:\dataset\\brats2021\\archive\\160_224_240\\224\\trainimage\\BraTS2021_01664_93.npy' # 替换为你的NumPy文件路径
# numpy_file_path_label = 'E:\dataset\\brats2021\\archive\\160_224_240\\224\\trainMask\\BraTS2021_01664_93.npy'
numpy_file_path_image = 'E:\dataset\\brats2020\\160_224_240\\224\\trainimage\\BraTS20_Training_342_100.npy' # 替换为你的NumPy文件路径
numpy_file_path_label = 'E:\dataset\\brats2020\\160_224_240\\224\\trainmask\\BraTS20_Training_342_100.npy'
image_array = np.load(numpy_file_path_image)
label_array = np.load(numpy_file_path_label)

# 假设image_data是一个形状为(240, 240)的二维数组，代表灰度图像
# 如果它是彩色图像，则它应该是一个形状为(240, 240, 3)的三维数组（例如RGB）
image_array = image_array[:, :, 1]

# 使用matplotlib显示图像


# 创建一个新的图形，并设置其大小
plt.figure(figsize=(10, 5))  # 可以根据需要调整图形大小

# 添加第一个子图并显示第一张图像
plt.subplot(1, 2, 1)  # 这里的参数表示1行2列，当前是第1个子图
plt.imshow(image_array,cmap='gray')
plt.title('Image 1')  # 为子图添加标题
plt.axis('off')  # 关闭坐标轴显示

# 添加第二个子图并显示第二张图像
plt.subplot(1, 2, 2)  # 这里的参数表示1行2列，当前是第2个子图
plt.imshow(label_array,cmap='gray')
plt.title('Image 2')  # 为子图添加标题
plt.axis('off')  # 关闭坐标轴显示

# 显示图形
plt.show()
pylab.show()
