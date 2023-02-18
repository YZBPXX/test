import os


def get_file_from_dir(dir_name, exts=('.arrow', )):
    files = []
    for parent, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.lower().endswith(exts):
                files.append(os.path.join(parent, filename))
    return files


if __name__ == '__main__':
    path = '/home/bo.zhu/Projects/aahq-dataset/raw/'
    path = '/data/storage1/public/bo.zhu/datasets/text2img/mj_yzb_0213/characters_upscaled/'
    path = '/data/storage1/public/bo.zhu/datasets/text2img/seeprettyfaces/'
    files = get_file_from_dir(path, ('.jpg', '.png'))
    # target_file = '/home/bo.zhu/Projects/aahq-dataset/'
    target_file = '/data/storage1/public/bo.zhu/datasets/text2img/train_0218.idx'
    with open(target_file, 'a') as f:
        for file in files:
            f.write(file + '\n')

