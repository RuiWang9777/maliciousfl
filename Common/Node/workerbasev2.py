import logging
import torch
import time
from abc import ABCMeta, abstractmethod

from Common.Utils.evaluate import evaluate_accuracy
# uploading weight
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local weights.
'''
class WorkerBaseV2(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, test_iter, config, optimizer):
        self.model = model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter

        self.config = config
        self.optimizer = optimizer

        # Accuracy record
        self.acc_record = [0]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._level_length = None
        self._weights_len = 0
        self._weights = None

    def get_weights(self):
        """ getting weights """
        return self._weights

    def set_weights(self, weights):
        """ setting weights """
        # try:
        #     if len(weights) < self._weights_len:
        #         raise Exception("weights length error!")
        # except Exception as e:
        #     logger.error(e)
        # else:
        self._weights = weights

    def train_step(self, x, y):
        """ Find the update gradient of each step in collaborative learning """
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._weights = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.grad.numel() + self._level_length[-1])
            self._gradients += param.grad.view(-1).numpy().tolist()

        self._grad_len = len(self._gradients)
        return loss.cpu().item(), y_hat

    def upgrade(self):
        """ Use the processed weights to update the model """
        # try:
        #     if len(self._gradients) != self._grad_len:
        #         raise Exception("gradients is wrong")
        # except Exception as e:
        #     logger.error(e)

        idx = 0
        for param in self.model.parameters():
            tmp = self._weights[self._level_length[idx]:self._level_length[idx + 1]]
            weights_re = torch.tensor(tmp, device=self.device)
            weights_re = weights_re.view(param.data.size())

            param.data = weights_re
            idx += 1

    def train(self):
        """ General local training methods """
        self.acc_record = [0]
        for epoch in range(self.config.num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in self.train_iter:
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(X)
                l = self.loss_func(y_hat, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

            test_acc = evaluate_accuracy(self.test_iter, self.model)
            self.eva_record += [test_acc]
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    def fl_train(self, times):
        self.acc_record = [0]
        counts = 0
        for epoch in range(self.config.num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in self.train_iter:
                counts += 1
                if (counts % times) != 0:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.model(X)
                    l = self.loss_func(y_hat, y)
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    train_l_sum += l.cpu().item()
                    train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                    n += y.shape[0]
                    batch_count += 1

                    continue

                loss, y_hat = self.train_step(X, y)
                self.update()
                self.upgrade()
                train_l_sum += loss
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

                if self.test_iter != None:
                    test_acc = evaluate_accuracy(self.test_iter, self.model)
                    self.acc_record += [test_acc]
                    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    def write_acc_record(self, fpath, info):
        s = ""
        for i in self.acc_record:
            s += str(i) + " "
        s += '\n'
        with open(fpath, 'a+') as f:
            f.write(info + '\n')
            f.write(s)
            f.write("" * 20)

    @abstractmethod
    def update(self):
        pass