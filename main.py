import json
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

train_tokens_file = 'train_tokens.json'
train_topics_file = 'train_topics.json'
test_tokens_file = 'test_tokens.json'
output_file = '10142510168.json'

k_docid = "docid"
k_topic = "topic"
k_cluster = "cluster"


def load_Vector(tokens_list):
    # 文档转换为向量矩阵
    vectorSet = []
    for data in tokens_list:
        data = data['tokenids']
        arr = [0 for i in range(0, 3508)]
        for i in data:
            arr[i] = arr[i] + 1
        vectorSet.append(arr)
    # print(vectorSet[1][2778])
    vectorSet = np.array(vectorSet)
    return vectorSet


def load_Center_Position():
    # 加载训练集的每一类的位置
    # 根据参考文件，有65个类别主题
    # 输出[1, 48, 81, 152, 245, 319, 416, 467, 571, 667, 726, 793, 871, 911, 993, 1069, 1124, 1171, 1250, 1300,
    pos = []
    type_num = 1
    pos.append(1)
    arr = load_array("train_topics.json")
    while (type_num <= 64):
        position = 0
        for data in arr:

            data = data['topic']
            if (data == type_num):
                pos.append(position)
                break
            position = position + 1

        type_num = type_num + 1
    pos.append(len(arr))
    return pos


def load_Means_Vector(vectorSet, pos):
    # 返回训练集每一类的均值向量
    means_Vector_set = []

    a = 0
    b = 1
    length = len(pos)
    while (b < length):
        means_Vector = [0 for i in range(0, 3508)]
        type_length = pos[b] - pos[a] + 1
        for i in range(pos[a] - 1, pos[b]):
            means_Vector = means_Vector + vectorSet[i]
        means_Vector = [(1 / type_length) * x for x in means_Vector]
        means_Vector_set.append(means_Vector)
        a = b
        b = b + 1

    return np.array(means_Vector_set)


def result_list(tokens_list, type_list):
    # 生成返回的最终列表
    length = len(tokens_list)
    result_list = []
    for i in range(0, length):
        doc_type = dict()
        docid = tokens_list[i]["docid"]
        cluster = type_list[i]
        doc_type["docid"] = docid
        doc_type["cluster"] = cluster
        result_list.append(doc_type)
    # print(result_list)
    return result_list


def execute_cluster(tokens_list):
    """
    输入一组文档列表，返回每条文档对应的聚类编号
    :param tokens_list: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'tokenids': list，每个元素为int，单词标识符
    :return: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'cluster': int，标识该文档的聚类编号
    """

    print(type(tokens_list), type(tokens_list[0]), list(tokens_list[0].items()))  # 仅用于验证数据格式
    #### 修改此处 ####
    # 加载训练集
    training_set = load_Vector(load_array(train_tokens_file))
    # 加载训练集的不同类别的标志位
    pos = load_Center_Position()
    # 计算参考类别中心向量集合
    means_vector_set = load_Means_Vector(training_set, pos)
    # 加载测试集
    vectorSet = load_Vector(tokens_list)
    # 训练数据
    k_means = KMeans(n_clusters=65, init=means_vector_set, n_init=10, max_iter=3000, n_jobs=5, tol=0.00001,
                     precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True, algorithm='auto').fit(training_set)
    # 预测数据
    type_list = list(k_means.predict(vectorSet))
    # 返回结果
    clusters_list = result_list(tokens_list, type_list)
    return clusters_list


""" 以下内容修改无效 """


def calculate_nmi(topics_list, clusters_list):
    id2topic = dict([(d[k_docid], d[k_topic]) for d in topics_list])
    id2cluster = dict([(d[k_docid], d[k_cluster]) for d in clusters_list])
    common_idset = set(id2topic.keys()).intersection(id2cluster.keys())
    if not len(common_idset) == len(topics_list) == len(clusters_list):
        print(len(common_idset), len(topics_list), len(clusters_list))
        print('length inconsistent, result invalid')
        return 0
    else:
        topic_cluster = [(id2topic[docid], id2cluster[docid]) for docid in common_idset]
        y_topic, y_cluster = list(zip(*topic_cluster))
        nmi = metrics.normalized_mutual_info_score(y_topic, y_cluster)
        print('nmi:{}'.format(round(nmi, 4)))
        return nmi


def dump_array(file, array):
    lines = [json.dumps(item, sort_keys=True) + '\n' for item in array]
    with open(file, 'w') as fp:
        fp.writelines(lines)


def load_array(file):
    with open(file, 'r') as fp:
        array = [json.loads(line.strip()) for line in fp.readlines()]
    return array


def clean_clusters_list(clusters_list):
    return [dict([(k, int(d[k])) for k in [k_docid, k_cluster]]) for d in clusters_list]


def evaluate_train_result():
    train_tokens_list = load_array(train_tokens_file)
    train_clusters_list = execute_cluster(train_tokens_list)
    train_topics_list = load_array(train_topics_file)
    calculate_nmi(train_topics_list, train_clusters_list)


def generate_test_result():
    test_tokens_list = load_array(test_tokens_file)
    test_clusters_list = execute_cluster(test_tokens_list)
    test_clusters_list = clean_clusters_list(test_clusters_list)
    dump_array(output_file, test_clusters_list)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    if args.train:
        evaluate_train_result()
    elif args.test:
        generate_test_result()
