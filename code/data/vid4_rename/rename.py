# -*- coding: utf-8 -*-            
# @Time    : 2023/3/21 14:41
# @Author  : Hong Chaming
import os


def re_filename(path):
    file_list = os.listdir(path)
    num = 85
    for file in file_list:
        used_filename, extension = os.path.splitext(file)
        new_file = f'lr_{num:02d}' + extension
        os.rename(file, new_file)
        print("文件%s重命名成功，新的文件名为：%s" % (path + file, path + new_file))
        num += 1


if __name__ == '__main__':
    path = os.getcwd()
    re_filename(path)


