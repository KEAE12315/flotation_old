

class Evaluation():
    """驻足点提取评价指标"""

    def __init__(self, df_predication, df_label) -> None:
        assert df_predication.shape[0] == df_label.shape[0], 'Data size mismatch'
        self.dfp = df_predication
        self.dfl = df_label
        self.initBasic()

    def setDfp(self, df_predication):
        self.dfp = df_predication
        self.initBasic()

    def initBasic(self):
        self.tp = self.TP()
        self.tn = self.TN()
        self.fp = self.FP()
        self.fn = self.FN()

    def TP(self):
        """True positive"""
        indexsPositive = self.dfp[self.dfp['type'] == 'activity'].index.tolist()
        tmp = self.dfl.loc[indexsPositive]
        return tmp[tmp['type'] == 'activity'].shape[0]

    def TN(self):
        """True negative"""
        indexsNegatve = self.dfp[self.dfp['type'] != 'activity'].index.tolist()
        tmp = self.dfl.loc[indexsNegatve]
        return tmp[tmp['type'] != 'activity'].shape[0]

    def FP(self):
        """False positive"""
        indexsPositive = self.dfp[self.dfp['type'] == 'activity'].index.tolist()
        tmp = self.dfl.loc[indexsPositive]
        return tmp[tmp['type'] != 'activity'].shape[0]

    def FN(self):
        """False negative"""
        indexsNegatve = self.dfp[self.dfp['type'] != 'activity'].index.tolist()
        tmp = self.dfl.loc[indexsNegatve]
        return tmp[tmp['type'] == 'activity'].shape[0]

    def returnHandler(func):
        def warpper(self):
            try:
                return func(self)
            except ZeroDivisionError:
                return -1
        return warpper

    @returnHandler
    def Acc(self):
        """精度: 预测对了多少"""
        return (self.tp + self.tn) / self.dfp.shape[0]

    @returnHandler
    def Precision(self):
        """查准率: 预测阳性中真阳性的比例"""
        return self.tp / (self.tp + self.fp)

    @returnHandler
    def Recall(self):
        """查全率: """
        return self.tp / (self.tp + self.fn)

    @returnHandler
    def FI(self):
        """F值: """
        precision = self.Precision()
        recall = self.Recall()
        return 2 * precision * recall / (precision + recall)

    def info(self):
        msg = f'total points: {self.dfp.shape[0]}\n' +\
            f'tp: {self.tp}\n' +\
            f'tn: {self.tn}\n' +\
            f'fp: {self.fp}\n' +\
            f'fn: {self.fn}\n\n' +\
            f'accuracy: {self.Acc():2.2%}\n' +\
            f'precision: {self.Precision():2.2%}\n' +\
            f'recall: {self.Recall():2.2%}\n' +\
            f'F index: {self.FI():.2}\n'
        print(msg)
        # return self.dfp.shape[0], self.tp, self.tn, self.fp, self.fn, self.Acc(), self.Precision(), self.Recall(), self.FI()


def convert(df):
    """列转换"""
    tmp = df.copy(deep=True)
    tmp = tmp[['type', 'pre1', 'pre2']]
    tmp.loc[tmp['type'] != 'activity', 'type'] = False
    tmp.loc[tmp['type'] == 'activity', 'type'] = True
    tmp.loc[tmp['pre1'] != 'activity', 'pre1'] = False
    tmp.loc[tmp['pre1'] == 'activity', 'pre1'] = True
    tmp.loc[tmp['pre2'] != 'activity', 'pre2'] = False
    tmp.loc[tmp['pre2'] == 'activity', 'pre2'] = True
    return tmp.type, tmp.pre1, tmp.pre2


class ConfusionMatrix2():
    """二分类混淆矩阵中的评价指标"""

    def __init__(self, predictions, labels) -> None:
        """
        Args:
        - prediction: 预测值, 为bool值的可迭代对象
        - labels: 真实值, 为bool值的可迭代对象
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        for prediction, label in zip(predictions, labels):
            if prediction and label:
                self.tp += 1
            elif prediction and not label:
                self.fp += 1
            elif not prediction and label:
                self.fn += 1
            elif not prediction and not label:
                self.tn += 1

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def returnHandler(self, func):
        def warpper(self):
            try:
                return func(self)
            except:
                return -1
        return warpper


class EvalCluster:
    def __init__(self, df_prediction, df_label) -> None:
        self.dfp = df_prediction
        self.dfl = df_label
        self.postiveNums = df_label[df_label.type == 'activity'].shape[0]

    def setDfp(self, df_prediction):
        self.dfp = df_prediction

    def overview(self):
        df = self.dfp
        df = df[df.clusterID != -1]
        indexs = df.index.to_list()
        df = self.dfl.loc[indexs]

        sampleSize = df.shape[0]
        positiveNums = df[df.type == 'activity'].shape[0]
        negativeNums = df[df.type != 'activity'].shape[0]
        positiveProp = positiveNums / self.postiveNums
        negativeProp = negativeNums / sampleSize

        return positiveProp, positiveNums, self.postiveNums

    # def detail(self):
    #     details = {}

    #     sampleTotal = 0
    #     positiveTotal = 0
    #     negativeTotal = 0
    #     for clusterID, df in dfp.groupby('clusterID'):
    #             indexs = df.index.to_list()
    #             dfr = dfl.loc[indexs]
    #             sampleSize = dfr.shape[0]
    #             positiveNums = dfr[dfr.type == 'activity'].shape[0]
    #             negativeNums = dfr[dfr.type != 'activity'].shape[0]
    #             positiveProp = positiveNums / sampleSize
    #             negativeProp = negativeNums / sampleSize

    #             sampleTotal = sampleTotal + sampleSize
    #             positiveTotal = positiveTotal + positiveNums
    #             negativeTotal = negativeTotal + negativeNums

    #             details[clusterID] = {
    #                 'sampleSize': sampleSize,
    #                 'positiveNums': positiveNums,
    #                 'negativeNums': negativeNums,
    #                 'positiveProp': positiveProp,
    #                 'negativeProp': negativeProp,
    #             }

    #         # return sampleTotal, positiveTotal, negativeTotal, details
    #         return positiveTotal / sampleTotal, negativeTotal / sampleTotal
