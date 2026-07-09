#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author      : Gao Jie
@date        : 2024-11-20
@file        : process_file.py
@description : process file functions
@version     : 1.0
"""

import os


def count_single_file(path):
    if not os.path.exists(path):
        return

    files = os.listdir(path)
    print(f'{path}: {len(files)}')

def count_split_files(dir):
    path_list = [
                 os.path.join(dir, 'eyelid'),
                 os.path.join(dir, 'single_eyelid'),
                 os.path.join(dir, 'double_eyelid'),
                 ]

    for path in path_list:
        count_single_file(path)

def count_files():
    file_list = [
        '../data/celeba',
        '../data/ffhq',
    ]

    for file in file_list:
        count_split_files(file)

#========================================================================

def check_split_files(dir):
    path_list = [
                 os.path.join(dir, 'eyelid'),
                 os.path.join(dir, 'single_eyelid'),
                 os.path.join(dir, 'double_eyelid'),
                 ]

    img_list = sorted(os.listdir(path_list[0]))
    for path in path_list[1:]:
        if not os.path.exists(path):
            continue
        print(f'{path}: {img_list == sorted(os.listdir(path))}')

def check_files():
    file_list = [
        '../data/celeba',
        '../data/ffhq',
    ]

    for file in file_list:
        check_split_files(file)

#========================================================================

def rename_split_files(dir):
    path_list = [
                 os.path.join(dir, 'eyelid'),
                 os.path.join(dir, 'single_eyelid'),
                 os.path.join(dir, 'double_eyelid'),
                 ]

    for path in path_list:
        new_path = path.replace('eyelid', 'eyelid_res')
        if os.path.exists(path):
            os.rename(path, new_path)

def rename_files():
    file_list = [
        '../data/celeba',
        '../data/ffhq',
    ]

    for file in file_list:
        rename_split_files(file)

def rename_suffix():
    file_dir = '/Users/bigo10295/Downloads/test_data/source_img_explore'
    for file in os.listdir(file_dir):
        if file.endswith('.jpeg'):
            os.rename(os.path.join(file_dir, file), os.path.join(file_dir, file.replace('.jpeg', '.jpg')))

#========================================================================

def select_intersection_files(path1, path2):
    file1 = set(os.listdir(path1))
    file2 = set(os.listdir(path2))
    print(f'before remove: file1: {len(file1)}, file2: {len(file2)}')

    intersection_file = file1.intersection(file2)

    remove_file1 = file1.difference(intersection_file)
    remove_file2 = file2.difference(intersection_file)

    for name in remove_file1:
        os.remove(os.path.join(path1, name))

    for name in remove_file2:
        os.remove(os.path.join(path2, name))

    print(f'after remove: file1: {len(os.listdir(path1))}, file2: {len(os.listdir(path2))}')


#========================================================================

if __name__ == '__main__':
    # src_dir = '../data/celeba'
    # count_single_file(src_dir)

    # count_split_files(src_dir)

    # count_files()

    # check_files()

    # rename_files()

    # select_intersection_files('../data/celeba/res1', '../data/celeba/res2')

    rename_suffix()
