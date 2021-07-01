import glob
import os
import os.path as osp
from PIL import Image

from torch.utils.data import DataLoader

from IPython import embed

def process_dir(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*'))
    # embed()
    ### 移除cifar10数据集
    img_paths.remove(dir_path + '/cifar-10-python.tar.gz')
    img_paths.remove(dir_path + '/cifar-10-batches-py')
    dataset = []
    i = 0
    ### 类名
    for img_path in img_paths:
        i += 1
        print("img_path:",img_path)

        target = img_path.split('/')[3].split('.')[0]
        class_name = img_path.split('/')[3]

        ### 创建目录
        base_dir = "../output/train/train" + str(i)
        print("base_dir:",base_dir)
        print("temp_dir",base_dir)
        # base_dir = osp.join(temp_dir,clas s_name)
        # base_dir = "../output/train/train" + str(i+1)
        if not osp.exists(base_dir):
            os.makedirs(base_dir)
        # imgs_dir = osp.join(base_dir,class_name)
        # embed()
        sub_img_paths = glob.glob(osp.join(img_path, "*.jpg"))
        ### 每个类别下的图像
        for sub_img_path in sub_img_paths:
            dataset.append((sub_img_path,target))
            sub_img_name = sub_img_path.split('/')[4]
            img = Image.open(sub_img_path)
            img.save(base_dir + "/" + sub_img_name)
    #
    #
    #
    # # pattern = re.compile(r'([-\d]+)_c(\d)')
    #
    # pid_container = set()
    # for img_path in img_paths:
    #     pid, _ = map(int, pattern.search(img_path).groups())
    #     if pid == -1: continue  # junk images are just ignored
    #     pid_container.add(pid)
    # pid2label = {pid: label for label, pid in enumerate(pid_container)}
    #
    # dataset = []
    # for img_path in img_paths:
    #     pid, camid = map(int, pattern.search(img_path).groups())
    #     if pid == -1: continue  # junk images are just ignored
    #     assert 0 <= pid <= 1501  # pid == 0 means background
    #     assert 1 <= camid <= 6
    #     camid -= 1  # index starts from 0
    #     if relabel: pid = pid2label[pid]
    #     dataset.append((img_path, pid, camid))
    #
    # num_pids = len(pid_container)
    # num_imgs = len(dataset)
    # return dataset, num_pids, num_imgs
    return dataset

if __name__ == "__main__":
    dir_path = "../data/train"
    dataset = process_dir(dir_path)
    print(dataset)