import torch
import torch.nn.functional as F

import json
import matplotlib
from utils.tools import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#from dataloader.Douban.dataset_clip import (trailer_multimodal_features,)
from dataloader.dataset_L_3 import trailer_multimodal_features
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ranking_loss(true_labels, predictions):
    """
    ranking_loss。
    """
    n_samples, n_labels = true_labels.shape

    # 初始化排名损失
    total_loss = 0

    # 遍历每个样本
    for i in range(n_samples):
        # 获取当前样本的真实标签和预测分数
        true_i = true_labels[i]
        pred_i = predictions[i]

        # 对预测分数进行排序，获取排名
        arg_sort_pred = np.argsort(-pred_i)  # 降序排序
        ranks = np.argsort(arg_sort_pred)  # 获取排名

        # 计算排名损失
        for j in range(n_labels):
            if true_i[j]:  # 如果标签是相关的
                # 计算排名高于当前标签的所有不相关标签的数量
                above_related = ranks[j + 1:][np.logical_not(true_i[arg_sort_pred[j + 1:]])]
                total_loss += len(above_related)

    # 归一化排名损失
    total_related_labels = np.sum(true_labels, axis=1)
    ranking_loss = total_loss / np.sum(total_related_labels)

    return ranking_loss


def coverage(true_labels, pred_scores):
    """
    计算Coverage指标。

    参数:
    true_labels: 二进制数组，形状为 (n_samples, n_labels)，表示真实标签。
    pred_scores: 实数数组，形状为 (n_samples, n_labels)，表示模型输出的标签分数。

    返回:
    coverage: Coverage指标的值。
    """
    n_samples, n_labels = true_labels.shape
    total_coverage = 0

    # 遍历每个实例
    for i in range(n_samples):
        # 获取真实标签和预测分数
        true_i = true_labels[i]
        pred_score_i = pred_scores[i]

        # 对预测分数进行降序排序，并获取排名
        sorted_indices = np.argsort(pred_score_i)[::-1]  # 从高到低排序

        # 找到覆盖所有真实相关标签所需的最小标签数量
        covered_labels = 0
        for rank, idx in enumerate(sorted_indices):
            if true_i[idx]:
                covered_labels += 1
                # 更新Coverage，取当前实例的最大覆盖排名
                if covered_labels == np.sum(true_i):
                    total_coverage += rank + 1  # rank从0开始，所以需要+1
                    break

    # 计算平均Coverage
    coverage = total_coverage / n_samples

    return coverage

dataloader_dict = {
    "trailer_multimodal_features": trailer_multimodal_features,
}
#+++++++++++++++++++++++++++++++++++++
def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float64: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)
#++++++++++++++++++++++++++++++++++++++++++++++++++
def test(args, config):
    dataloader = dataloader_dict[args.dataloader_name]

    testdata = dataloader(phase='test',
                          data_dir=config['data_dir'],
                          feature_dir=config['feature_dir'],
                          visual_feature_dir=config['visual_feature_dir'],
                          visual_feature_file=config['visual_feature_version'] + "_test.pkl" if config[
                              'visual_feature_version'] else None,
                          cau_visual=config['cau_visual'],
                          audio_feature_dir=config['audio_feature_dir'],
                          audio_feature_file=config['audio_feature_file'],
                          cau_audio=config['cau_audio'],
                          text_feature_dir=config['text_feature_dir'],
                          text_token_file=config['text_token_file'],
                          text_feature_file=config['text_feature_file'],
                          cau_text=config['cau_text'],
                          meta_glo_file=config['meta_glo_file'],
                          meta_loc_file=config['meta_loc_file'],
                          num_classes=config["num_categories"],
                          )

    test_loader = torch.utils.data.DataLoader(testdata, batch_size=config['batch_size'], shuffle=False,
                                              num_workers=config['n_cpu'])
    model = torch.load(os.path.join(args.model_path, "epoch-best.pkl"))
    return_predictions = []
    return_labels = []

    with torch.no_grad():
        model.eval()  # model.eavl() fix the BN and Dropout
        test_loss = 0
        for batch_idx, sample in enumerate(test_loader):
            movie_id, label = sample['movie_id'],sample['label_onehot']
            f_one = {}
            f_one['visual_feature'] = sample['visual_feature'].to(device).type(FloatTensor)
            # f_one['text_feature'] = sample['text_feature'].to(device).type(FloatTensor)
            # f_one['audio_feature'] = sample['audio_feature'].to(device).type(FloatTensor)
            f_one['cau_visual_feature'] = sample['cau_visual_feature'].to(device).type(FloatTensor)
            f_one['VTM_visual_feature'] = sample['VTM_visual_feature'].to(device).type(FloatTensor)
            f_one['topic_feature'] = sample['topic_feature'].to(device).type(FloatTensor)
            f_one['VTM_topic_feature'] = sample['VTM_topic_feature'].to(device).type(FloatTensor)
            label = label.type(LongTensor)

            type_index = torch.cat((torch.zeros(f_one['visual_feature'].shape[-1], dtype=torch.int64),
                                    torch.ones(f_one['cau_visual_feature'].shape[-1], dtype=torch.int64)), dim=0).to(device)
            pe = generate_positional_encoding(args.hidden_dim, args.ori_len + args.cau_len).to(device)
            #output = model(f_one, type_index, pe, args.lambda_f)
            output = model(f_one, type_index, pe, args.lambda_f, args.lambda_m)
            test_loss += F.binary_cross_entropy_with_logits(output, label.float())
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()

            return_predictions.append(predicted_np)
            return_labels.append(label_np)

    return np.concatenate(return_predictions), np.concatenate(return_labels)

def test_main(args):
    config_file = os.path.join(args.model_path, "train_info.json")
    with open(config_file, 'r') as f:
        config = json.load(f)

    outputs, labels = test(args, config)
    outputs_sigmoid = sigmoid(outputs)
    print("outputs shape:", outputs.shape, labels.shape)
    average_choice = ['micro', 'samples', 'weighted', 'macro']
    mAP = {}
    precision = {}
    recall = {}
    for i in range(4):
        mAP[average_choice[i]] = average_precision_score(labels[:, :], outputs_sigmoid[:, :], average=average_choice[i])
        precision[average_choice[i]] = precision_score(labels, outputs_sigmoid >= 0.5, average=average_choice[i])
        recall[average_choice[i]] = recall_score(labels, outputs_sigmoid >= 0.5, average=average_choice[i])
    coverageloss = coverage_error(labels, outputs_sigmoid)
    rl = label_ranking_loss(labels, outputs_sigmoid)
    print("recall-macro: ", recall['macro'])
    print("precision-macro: ", precision['macro'])
    print("mAP-macro: ", mAP['macro'])
    print("recall-micro: ", recall['micro'])
    print("precision-micro: ", precision['micro'])
    print("mAP-micro: ", mAP['micro'])
    print("coverage: ", coverageloss)
    print("ranking_loss: ", rl)

    # curve_precision, curve_recall, curve_thresholds = precision_recall_curve(labels, outputs_sigmoid)

    """ calculate precision and recall for each category outputs@0.5"""
    outputs_prediction = outputs_sigmoid >= 0.5

    precisions = {}
    recalls = {}
    for j in range(outputs.shape[1]):
        ptotal = 0
        pcorrect = 0
        rtotal = 0
        rcorrect = 0
        for i in range(outputs.shape[0]):
            if outputs_prediction[i][j]:
                ptotal += 1
                if labels[i][j]:
                    pcorrect += 1

            if labels[i][j]:
                rtotal += 1
                if outputs_prediction[i][j]:
                    rcorrect += 1

        # assert rtotal!=0
        # assert ptotal!=0
        # print(pcorrect/ptotal)
        precisions[j] = -0.01 if ptotal == 0 else pcorrect / ptotal
        recalls[j] = -0.01 if rtotal == 0 else rcorrect / rtotal

    if args.save_results:
        test_results = {}
        test_results['best epoch'] = config['best_epoch']
        test_results['mAP'] = mAP
        test_results['precision'] = precision
        test_results['recall'] = recall
        test_results['precision per genre'] = precisions
        test_results['recall per genre'] = recalls
        with open(os.path.join(args.model_path, 'testing_results.json'), 'w') as ft:
            json.dump(test_results, ft, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='logs/trailer_clip', help='path to trained model')
    parser.add_argument('--dataloader_name', type=str, default="trailer_clipfeat", help="specify a dataloader")
    parser.add_argument('--save_results', type=bool, default=True, help='dump test results in a json file')
    args = parser.parse_args()
    print(args)
    test_main(args)
