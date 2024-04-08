import pandas as pd
import torch
import sys
import numpy as np
# from scipy.stats import mode
# from scipy.spatial.distance import squareform
import glob

# 开启GPU加速
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')


class KnnDtw(object):
    def __init__(self, n_neighbors=5, max_warping_window=5000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x):
        self.x = x

    def _dtw_distance(self, ts_a, ts_b, bsf_score, d=lambda x, y: torch.sqrt(torch.sum((x - y) ** 2))):
        M, N = ts_a.shape[0], ts_b.shape[0]
        cost = torch.full((M, N), sys.maxsize, dtype=torch.float32).to(device)

        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
                if cost[i, j] > bsf_score:
                    return np.inf

        return cost[-1, -1].item()

    def envelope(self, this_tensor):
        '''
        计算上下包络线

        :param this_tensor: 维度为[n,p]的tensor，一个多元时间序列，其中n为时间序列长度，p为指标数量
        :return: tensor上下包络线max_A和min_A,均为多元时间序列
        '''
        # 使用 unfold 函数创建滑动窗口
        n = this_tensor.shape[0]
        # unfold会生成n-2*"fai"个窗口，为了保证长度相等，增加2*“fai”个数据
        t_dup = torch.cat([this_tensor[:self.max_warping_window, :], this_tensor, this_tensor[n - self.max_warping_window:, :]], dim=0)
        window_view = t_dup.unfold(0, 2 * self.max_warping_window + 1, 1)

        # 分别计算最大值和最小值
        max_this_tensor = torch.max(window_view, dim=2)[0]
        min_this_tensor = torch.min(window_view, dim=2)[0]

        return max_this_tensor, min_this_tensor

    def _dist_list(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : list of dataframes [n_timepoints, n_features] test_set

        y : list of dataframes [n_samples, n_features] train_set

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        # Compute the distance matrix
        dm_count = 0

        x_s = len(x)
        y_s = len(y)
        dm_size = x_s * y_s
        anomaly_score = []

        p = ProgressBar(dm_size)

        for i in range(0, x_s):
            this_x = x[i].clone()
            bsf_distance = np.inf
            upper_envelope, lower_envelope = self.envelope(this_x)
            for j in range(0, y_s):
                this_y = y[j].clone()
                lb_kim = torch.norm(this_x[0] - this_y[0]) + torch.norm(this_x[-1] - this_y[-1])
                if lb_kim < bsf_distance:
                    lb_keogh = (torch.sum(torch.pow((this_y - upper_envelope)[this_y > upper_envelope], 2)) +
                                torch.sum(torch.pow((this_y - lower_envelope)[this_y < lower_envelope], 2))).item()
                    if lb_keogh < bsf_distance:
                        this_yy = this_y.clone()
                        this_yy = torch.where(this_yy > upper_envelope, upper_envelope, this_yy).clone()
                        this_yy = torch.where(this_yy < lower_envelope, lower_envelope, this_yy).clone()
                        lbi_u_envelope, lbi_l_envelope = self.envelope(this_yy)
                        lb_i_keogh = (torch.sum(torch.pow((this_x - lbi_u_envelope)[this_x > lbi_u_envelope], 2)) +
                                      torch.sum(torch.pow((this_x - lbi_l_envelope)[this_x < lbi_l_envelope], 2))).item()
                        if lb_keogh + lb_i_keogh < bsf_distance:
                            dtw_distance_score = self._dtw_distance(this_x, this_y, bsf_distance)
                        else:
                            dtw_distance_score = np.inf
                            print('skip3!')
                    else:
                        dtw_distance_score = np.inf
                        print('skip2!')
                else:
                    dtw_distance_score = np.inf
                    print('skip1!')

                if (dtw_distance_score < bsf_distance) and (dtw_distance_score != 0):
                    bsf_distance = dtw_distance_score

                # Update progress bar
                dm_count += 1
                p.animate(dm_count)

            anomaly_score.append(bsf_distance)

        return anomaly_score

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : list of array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dll = self._dist_list(x, self.x)

        return dll


class ProgressBar:
    """This progress bar was taken from PYMC
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.percent_done_sign = -1
        self.width = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def __str__(self):
        return str(self.prog_bar)

    def animate_ipython(self, iter):
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        if percent_done == self.percent_done_sign:
            pass
        else:
            all_full = self.width - 2
            num_hashes = int(round((percent_done / 100.0) * all_full))
            self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
            pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
            pct_string = '%d%%' % percent_done
            self.prog_bar = self.prog_bar[0:pct_place] + (pct_string + self.prog_bar[pct_place + len(pct_string):])
            print(str(self.prog_bar))
            self.percent_done_sign = percent_done


def sequence_extension(this_list_tensor: torch.tensor, this_len):
    new_list_tensor = list()
    extended_row = torch.zeros((1, this_list_tensor[0].shape[1]), dtype=torch.float32).to(device)
    for this_tensor in this_list_tensor:
        dif_num = this_len - this_tensor.shape[0]
        extended_rows = extended_row.repeat(dif_num, 1)
        tmp_this_tensor = torch.cat([this_tensor, extended_rows])
        new_list_tensor.append(tmp_this_tensor)
    return new_list_tensor


# =================================数据运算============================================

stock_index = '002479'
data_path_1 = './dfs/' + stock_index

file = glob.glob(data_path_1 + "/*.csv")

snap_time = np.array([92500000, 94500000, 100000000, 101500000, 103000000, 104500000, 110000000, 111500000, 113000000, 131500000, 133000000, 134500000, 140000000, 141500000, 143000000, 144500000, 150000000])

# 分离出训练集
file_train = file[:-2]
file_test = [file[-1]]      # 只有一个的时候加[]
# file_train = file[:-2]    # 000958


def read_list_file(file_list):
    dl = []
    for z in range(len(snap_time) - 1):
        dl.append([])
    for f in file_list:
        this_df = pd.read_csv(f, dtype='float32', header=0)

        # 涨跌停时没有对手方价格，因此‘价格冲击’为空，还有一些inf将其填充为0
        this_df['ImediPrImpac'][np.isinf(this_df['ImediPrImpac'])] = np.nan
        this_df['ImediPrImpac'].fillna(0.0, inplace=True)
        this_df['LongtPrImpac'][np.isinf(this_df['ImediPrImpac'])] = np.nan
        this_df['LongtPrImpac'].fillna(0.0, inplace=True)
        this_df.drop(this_df.tail(1).index, inplace=True)

        this_df = this_df[this_df['time'] != 150000000].copy()
        for z in range(len(snap_time) - 1):
            snap_df = this_df[(this_df['time'] >= snap_time[z]) & (this_df['time'] < snap_time[z + 1])]
            snap_df = snap_df[['volume', 'volume_dva', 'volume_period_sum', 'rv', 'ImediPrImpac', 'LongtPrImpac', 'duration']].copy()
            snap_np = torch.tensor(np.array(snap_df)).to(device)
            dl[z].append(snap_np)

    return dl


def test(train_data, test_data):
    test_df = pd.DataFrame()
    for k in range(len(snap_time) - 1):
        x_train = train_data[k]
        x_test = test_data[k]
        m = KnnDtw(n_neighbors=1, max_warping_window=10)
        # 在这一步先取出对应时间段的数据，fit和train仅输入预测的时间段内的dataframe

        len_list = [len(x) for x in x_train]
        len_list = len_list + [len(x) for x in x_test]
        max_len = max(len_list) + 1

        x_train = sequence_extension(x_train, max_len)
        x_test = sequence_extension(x_test, max_len)

        m.fit(x_train)

        proba = m.predict(x_test)

        test_df[f'{snap_time[k]/100000}-{snap_time[k + 1]/100000}'] = pd.DataFrame(proba)

        # proba.to_csv(r'./dfs/train' + '+' + stock_index + '+' + f'start={snap_time[k]}.csv', index=False, sep=',')
        print(f'完成{snap_time[k]}至{snap_time[k + 1]}的评分。')
    return test_df


train_dl = read_list_file(file_train)
test_dl = read_list_file(file_test)
# train_result = test(train_dl, train_dl)
# train_result.to_csv(r'./dfs/train' + '+' + stock_index + '.csv', index=False, sep=',')

test_result = test(train_dl, test_dl)
test_result.to_csv(r'./dfs/test23' + '+' + stock_index + '.csv', index=False, sep=',')






