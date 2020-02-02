import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol

model = {}
right = 0
false = 0


class MRNaiveBayesTrain(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRNaiveBayesTrain, self).__init__(*args, **kwargs)
        self.size = 0  # 训练集大小 the size of the training set

    '''
    Naive Bayes train set。take the input，output((category, features),(features, numbers))，or
    ((category, feature),(mean, std))（for continuous variable）。
    '''

    def configure_args(self):
        '''
        input args。including the continuous vairbales columns，separated by “,”
        '''
        super(MRNaiveBayesTrain, self).configure_args()
        self.add_passthru_arg("--continuous_features",
                              type=str,
                              help="type feature numbers that are continuous, use ',' to separate")

    def load_args(self, args):
        '''
        read data, get the column number of continuous features 将输入的连续特征编号记录下来。
        '''
        super(MRNaiveBayesTrain, self).load_args(args)
        if self.options.continuous_features is not None:
            self.continuous = []
            temp = self.options.continuous_features.split(',')
            for num in temp:
                try:
                    num = int(num)
                except:
                    # 输入的不是整数，报错。 if not continuous, output error
                    self.option_parser.error("The continuous features number you type in are not integer")
                self.continuous.append(num)
        else:
            self.continuous = []

    def steps(self):
        return ([MRStep(mapper=self.mapper, reducer=self.reducer)])

    def mapper(self, _, line):
        '''
        Mapper。get every line of the training set, get the features, output ((category, num of the feature),feature value)
        finally output((category,'all'),1) to count the total number.

        接收训练集每一行，抽取出其的特征集，输出((类别, 特征编号), 特征取值)。
        最后再输出((类别, 'all'), 1)，用于计算总数。
        '''
        feature = line.split(',')
        self.size += 1
        for i in range(len(feature) - 1):
            yield (feature[len(feature) - 1], i), (feature[i])
        yield (feature[len(feature) - 1], 'all'), 1

    def reducer(self, label, features):
        '''
        Reducer
        Count the number of features of every category, and the number of every category

        函数。统计每个类别下所有特征取值的数量，以及每个类别的数量。
        '''
        if (label[1] == 'all'):
            yield (label[0], label[1]), sum(features)
        elif (label[1] in self.continuous):
            # 若为连续特征，计算其均值和方差并yield, if continous, calculate mean and std,yield
            temp = []
            for feature in features:
                temp.append(int(feature))
            yield (label[0], label[1]), (np.mean(temp), np.std(temp))
        else:
            # 若非连续，则统计每个特征取值的数量, if not continous, count the number of every feature
            count = collections.Counter(features)
            for key in count:
                yield (label[0], label[1]), (key, count[key])


if (__name__ == '__main__'):
    MRNaiveBayesTrain.run()
