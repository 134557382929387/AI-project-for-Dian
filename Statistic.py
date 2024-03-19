class Statis:#指标统计函数
    def Accuracy(self, outputs, targets):
        correct = (outputs.argmax(1) == targets).sum().item()
        accuracy = correct / len(targets)
        return accuracy

    def Recall(self, outputs, targets):
        recall = 0
        for i in range(10):
            positive = (targets == i).sum().item()
            correct_prediction = ((outputs == i) & (targets == i)).sum().item()
            if positive != 0:
                rate = correct_prediction / positive
                recall += rate
        return recall / 10  # 取平均值作为最终结果

    def Precision(self, outputs, targets):
        precision = 0
        for i in range(10):
            positive = (outputs == i).sum().item()
            correct_prediction = ((outputs == i) & (targets == i)).sum().item()
            if positive != 0:
                rate = correct_prediction / positive
                precision += rate
        return precision / 10  # 取平均值作为最终结果

    def F1(self, outputs, targets):
        p = self.Precision(outputs, targets)
        r = self.Recall(outputs, targets)
        if p + r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)
