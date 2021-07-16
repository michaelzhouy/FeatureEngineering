# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 8:12 下午
# @Author  : Michael Zhouy
import pyhdfs


def get_hdfs_client(host='10.21.3.10', port=9000, username='hadoop'):
    """hdfs客户端"""
    try:
        return pyhdfs.HdfsClient(hosts=f'{host},{port}', user_name=f'{username}')
    except Exception as e:
        raise RuntimeError(f'get_hdfs_client: {e}')


def put_data(local_path, dt_part, area_part, y_part, is_x=True):
    """
    上传数据
    Args:
        local_path: 本地txt路径
        dt_part: 采样时间戳
        area_part: 采样区域编码
        y_part: 采样点y坐标，该y点下所有x，所有z，所有帧数据为一个txt

    Returns:

    """
    pyhdfs_client = get_hdfs_client()

    db_name = 'writing_height'
    tb_name = 'x_test' if is_x else 'y_test'

    # 取年月日即可
    dt_part = str(dt_part)[:10]

    # hdfs路径
    hdfs_dir = f'/user/hive/warehouse/{db_name}.db/{tb_name}/dt_part={dt_part}/area_part={area_part}/y_part={y_part}/data.txt'

    # 上传
    pyhdfs_client.copy_from_local(local_path, hdfs_dir)


def delete_partition(hdfs_dir):
    pyhdfs_client = get_hdfs_client()
    if pyhdfs_client.exists(hdfs_dir):
        file_list = pyhdfs_client.listdir(hdfs_dir)
        for file in file_list:
            hdfs_file_path = hdfs_dir + f'/{file}'
            print(f'delete {hdfs_file_path}')
            pyhdfs_client.delete(hdfs_file_path)


def alter_partition(hdfs_dir):
    pyhdfs_client = get_hdfs_client()
    if pyhdfs_client.exists(hdfs_dir):
        file_list = pyhdfs_client.listdir(hdfs_dir)
        for file in file_list:
            hdfs_file_path = hdfs_dir + f'/{file}'
            print(f'delete {hdfs_file_path}')
            pyhdfs_client.delete(hdfs_file_path)


if __name__ == '__main__':
    # 参数
    local_path = './test.txt'  # txt本地路径
    dt_part = '2021-07-01 10:00:00'  # 采样时间戳
    area_part = 1                    # 采样区域编码
    y_part = 1                       # 采样点y坐标，该y点下所有x，所有z，所有帧数据为一个txt

    # 上传操作
    put_data(local_path, dt_part, area_part, y_part)
