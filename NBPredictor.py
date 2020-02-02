import pandas as pd
import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import NaiveBayes
import os

current = os.getcwd()  # 当前路径 current path
true = 0  # 预测成功数量 the number of good prediction
false = 0  # 预测失败数量 the number of false prediction


class MRNaiveBayesTest(MRJob):

    def configure_args(self):
        '''
        input args, including the num of continuous feature, separated by ',', and the filepath to save the model.(the output of the result of NBTrain)
        输入args。包括连续特征的编号，用“,”隔开，以及model的路径（即NBTrain的输出）
        '''
        super(MRNaiveBayesTest, self).configure_args()
        self.add_passthru_arg("--continuous_features",
                              type=str,
                              help="type feature numbers that are continuous, use ',' to separate")
        self.add_passthru_arg("--model",
                              type=str,
                              help="the model path")

    def load_args(self, args):
        '''
        根据输入读取数据。
        get the input
        '''
        super(MRNaiveBayesTest, self).load_args(args)
        if self.options.continuous_features is not None:
            self.continuous = []
            temp = self.options.continuous_features.split(',')
            for num in temp:
                try:
                    num = int(num)
                except:
                    self.option_parser.error("The continuous features number you type in are not integer")
                self.continuous.append(num)

        # 读取model get the model
        if self.options.model is None:
            self.option_parser.error("please type the path to the model")
        else:
            self.model = {}  # 记录每个类别下所有特征取值的数量 count the number of features for each category
            self.total = {}  # 记录每个类别的总数 count the number of each distribution
            job = NaiveBayes.MRNaiveBayesTrain()
            with open(current + '/' + self.options.model, encoding='utf-8') as src:
                for line in src:
                    try:
                        # 该行不是'all'行，读取该类别下该特征下该特征取值的数量,
                        # if the line is not all, take the number of the features for this category
                        (cat, feature), (key, num) = job.parse_output_line(line.encode())
                    except:
                        # 该行是'all'行，读取该类别的总数量 if it is 'all', get the number of total features
                        (cat, _), num = job.parse_output_line(line.encode())
                        self.total[cat] = num
                        continue
                    if (cat not in self.model):
                        # 若该类别不在model中，建立该类别
                        #if this category not in the model, establish this category
                        self.model[cat] = {}
                    if (feature not in self.model[cat]):
                        # 若该特征不在model[cat]中，建立该特征
                        #if this feature not in model[cat], establish this feature
                        self.model[cat][feature] = {}
                    self.model[cat][feature][key] = num  # 记录数量 count the number

    def steps(self):
        return ([MRStep(mapper=self.mapper, reducer=self.reducer)])

    def __init__(self, *args, **kwargs):
        super(MRNaiveBayesTest, self).__init__(*args, **kwargs)

    def mapper(self, _, line):
        '''
        Mapper
        get every line of the training set, calculate the prob for each categoty, take the maximum
        compare with the real value, if same , output (true,1) else, output (false,1)

        函数，接收测试集每一行，并计算每个类型下该特征取值的概率，预测出最大可能的类型
        并与真实类型比较，若相同则输出(true, 1)，否则输出(false, 1)
        '''
        line = line[:-1]
        feature = line.split(',')
        res = {}  # 每个类型的概率, the prob for each category
        for key in self.model:
            # 对每个类型 for each category
            res[key] = self.total[key]  # initial value if the prob a prioi
            # 赋初值为该类型的先验概率
            for i in range(len(feature) - 1):
                if i not in self.continuous:
                    # 若为非连续特征 if not continous
                    try:
                        # 乘以条件概率 mutiplier conditional prob
                        res[key] *= self.model[key][i][feature[i]] / self.total[key]
                    except:
                        # 若model中不存在该类型的该特征取值，则直接赋0 if this value does not exist in the model, give 0
                        res[key] = 0
                else:
                    # 若为连续特征 if continous
                    mu = list(self.model[key][i].keys())[0]
                    sigma = self.model[key][i][mu]
                    # 乘以条件概率 multiplier conditional prob
                    res[key] *= stats.norm.pdf(int(feature[i]), mu, sigma)
        predict = max(res, key=res.get)  # 预测类别 predict the category
        if (predict == feature[len(feature) - 1]):
            # 若预测正确 if true
            yield 'true', 1
        else:
            yield 'false', 1

    def reducer(self, label, num):
        '''
        Reducer, count the number of true and false
        函数。统计预测正确和预测错误的测试样例数量。
        '''
        if False: yield
        if (label == 'true'):
            global true
            true = sum(num)
        else:
            global false
            false = sum(num)


if __name__ == '__main__':
    MRNaiveBayesTest.run()
    print("Accuary:" + str(true / (true + false) * 100) + "%")
