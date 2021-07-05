import scipy.io as sio
from PIL import Image
import os
import os.path as osp
from IPython import embed

mat_train_path = '../data/train_list.mat'
mat_test_path = '../data/test_list.mat'

train_list = sio.loadmat(mat_train_path)
test_list = sio.loadmat(mat_test_path)

# embed()

# 训练集
train_image_paths = train_list['file_list']# 真实路径为train_list['file_list'][i][0][0]
# 测试集
test_image_paths = test_list['file_list']# 真实路径为train_list['file_list'][i][0][0]

print("------ start splitting ------")
i = 1
for train_image_path in train_image_paths:
    print("image:",i)
    i += 1
    # 获取路径
    image_path_ = train_image_path[0][0]
    ### 获取类名
    image_class = image_path_.split('/')[0]
    image_name = image_path_.split('/')[1]
    real_image_path = "/home/gaoziqiang/resource/dataset/Stanford_Dogs_Dataset/images/Images/" + image_path_
    # 读取图像
    image = Image.open(real_image_path)
    image = image.convert('RGB')
    # 存储路径
    class_path = '../output/stanford_dogs_splited/' + image_class
    if not osp.exists(class_path):
        os.makedirs(class_path)
    new_image_path = class_path + '/' + image_name
    # embed()
    image.save(new_image_path)

print("------ end ------")
