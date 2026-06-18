#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split, KFold

class Dataset:
    def generate_data(self, n_samples, outlier_fraction, random_state=None):
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        n_inliers = int((1. - outlier_fraction)*n_samples)
        n_outliers = int(outlier_fraction*n_samples)
    
        ## normal dataset => sampling from normal distribution
        n_left = n_inliers // 2
        n_right = n_inliers - n_left
        X1 = 0.5 * rng.randn(n_left,2) + 0.01
        X2 = 0.5 * rng.randn(n_right,2) - 0.01
    
        X = np.r_[X1,X2]
    
        ## abnormal dataset => sampling from abnormal dataset
        offset = rng.randint(low=1, high=10)
        X = np.r_[X,rng.uniform(low=-offset, high=offset, size=(n_outliers, 2))]
    
        y = np.zeros(n_samples, dtype=int)
        y[-n_outliers:] = 1
    
        split_seed = 123 if random_state is None else random_state
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=split_seed)
        return X_train, X_test, y_train, y_test

    def generate_overlap_data(
        self,
        n_samples,
        outlier_fraction,
        separation=1.0,
        normal_std=0.5,
        anomaly_std=0.6,
        random_state=None,
    ):
        """Generate harder synthetic OCC data with ambiguous boundaries.

        Normal samples are centered around the origin. Anomalies are sampled from
        local Gaussian clouds whose centers lie on a ring with radius
        ``separation``. Smaller separation or larger anomaly_std increases class
        overlap and makes the one-class boundary less clear.
        """
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        n_inliers = int((1.0 - outlier_fraction) * n_samples)
        n_outliers = n_samples - n_inliers

        X_normal = rng.normal(loc=0.0, scale=normal_std, size=(n_inliers, 2))

        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_outliers)
        centers = np.column_stack([np.cos(angles), np.sin(angles)]) * separation
        X_anomaly = centers + rng.normal(loc=0.0, scale=anomaly_std, size=(n_outliers, 2))

        X = np.r_[X_normal, X_anomaly]
        y = np.r_[np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)]

        split_seed = 123 if random_state is None else random_state
        return train_test_split(X, y, stratify=y, test_size=0.2, random_state=split_seed)
    
    def dataset(
        self,
        n_samples,
        outlier_fraction,
        n_repeats=100,
        generator='original',
        random_state=None,
        **kwargs,
    ):
        X_train_dataset = [None for _ in range(n_repeats)]
        y_train_dataset = [None for _ in range(n_repeats)]

        X_test_dataset = [None for _ in range(n_repeats)]
        y_test_dataset = [None for _ in range(n_repeats)]

    ## make total n_repeats datasets
        for i in range(n_repeats):
            seed = None if random_state is None else random_state + i
            if generator == 'original':
                X_train, X_test, y_train, y_test = self.generate_data(
                    n_samples,
                    outlier_fraction,
                    random_state=seed,
                )
            elif generator in ('overlap', 'ambiguous', 'boundary'):
                X_train, X_test, y_train, y_test = self.generate_overlap_data(
                    n_samples,
                    outlier_fraction,
                    random_state=seed,
                    **kwargs,
                )
            else:
                raise ValueError("generator must be one of: 'original', 'overlap', 'ambiguous', 'boundary'")
    
            X_train_dataset[i] = X_train
            y_train_dataset[i] = y_train
    
            X_test_dataset[i] = X_test
            y_test_dataset[i] = y_test
    
        return X_train_dataset, y_train_dataset, X_test_dataset, y_test_dataset 

    def get_target_label_idx(self, labels, target):
        return np.argwhere(np.isin(labels, target)).flatten().tolist()
    
    def train_valid_set(self, X, y, cv):
        kf = KFold(n_splits=cv, shuffle=True, random_state=123)
        
        x_train_folds = []; x_val_folds = []; y_train_folds = []; y_val_folds = []

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            
            normal_index = self.get_target_label_idx(y_train, target=0)
            X_train, y_train = X_train[normal_index], y_train[normal_index]
            X_valid, y_valid = X_valid, y_valid
            
            x_train_folds.append(X_train); x_val_folds.append(X_valid); y_train_folds.append(y_train); y_val_folds.append(y_valid)
        
        return x_train_folds, x_val_folds, y_train_folds, y_val_folds
    

# if __name__ == "__main__":
#     # 파라미터 설정
#     N_SAMPLES = 1000          # 전체 샘플 개수
#     OUTLIER_FRACTION = 0.2    # 이상치 비율 (10%)

#     print("=== 데이터셋 생성 시작 (총 100개 세트) ===")
#     ds = Dataset()
    
#     # 100개의 데이터셋 일괄 생성
#     X_train_ds, y_train_ds, X_test_ds, y_test_ds = ds.dataset(
#         n_samples=N_SAMPLES, 
#         outlier_fraction=OUTLIER_FRACTION
#     )
    
#     # 다루기 쉽게 Numpy 배열로 변환
#     X_train_arr = np.array(X_train_ds)
#     y_train_arr = np.array(y_train_ds)
#     X_test_arr = np.array(X_test_ds)
#     y_test_arr = np.array(y_test_ds)

#     print("\n[생성 완료 - 전체 데이터 쉐이프 확인]")
#     print(f"학습용(Train) 데이터 : {X_train_arr.shape}")  # (100, 800, 2)
#     print(f"학습용(Train) 라벨   : {y_train_arr.shape}")  # (100, 800)
#     print(f"테스트용(Test) 데이터: {X_test_arr.shape}")  # (100, 200, 2)
#     print(f"테스트용(Test) 라벨  : {y_test_arr.shape}")  # (100, 200)

#     # -------------------------------------------------------------
#     # 실제 모델 학습에 활용할 때 (예: 0번째 세트 꺼내 쓰기)
#     # -------------------------------------------------------------
#     X_train_0 = X_train_arr[0]
#     y_train_0 = y_train_arr[0]
#     X_test_0 = X_test_arr[0]
#     y_test_0 = y_test_arr[0]

#     print("\n[첫 번째 세트(Index 0) 구성 세부 확인]")
#     print(f"Train 총 {len(y_train_0)}개 중 -> 정상(0): {np.sum(y_train_0 == 0)}개, 이상치(1): {np.sum(y_train_0 == 1)}개")
#     print(f"Test  총 {len(y_test_0)}개 중 -> 정상(0): {np.sum(y_test_0 == 0)}개, 이상치(1): {np.sum(y_test_0 == 1)}개")
