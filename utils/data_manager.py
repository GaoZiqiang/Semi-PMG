import glob
import os.path as osp
from torch.utils.data import DataLoader

from IPython import embed

def process_dir(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*'))
    # embed()
    img_paths.remove(dir_path + '/cifar-10-python.tar.gz')
    img_paths.remove(dir_path + '/cifar-10-batches-py')
    dataset = []
    for img_path in img_paths:
        target = img_path.split('/')[3].split('.')[0]
        # embed()
        sub_img_paths = glob.glob(osp.join(img_path, "*.jpg"))

        for sub_img_path in sub_img_paths:
            dataset.append((sub_img_path,target))
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