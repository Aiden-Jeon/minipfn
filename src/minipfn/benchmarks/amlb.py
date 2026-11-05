"""AMLB·OpenML-CTR23 benchmark (논문 Fig. 4, Supplementary Tables 1·2).

논문은 샘플 10,000개, feature 500개, 클래스 10개 이하인 분류 29개와 회귀 28개를 썼다. 데이터셋마다
OpenML task가 정한 10-fold 교차검증(학습 90%, 테스트 10%)의 평균 점수를 낸다. 분류는 ROC AUC
(one-vs-rest), 회귀는 negative RMSE이다.

`PAPER_CLASSIFICATION`, `PAPER_REGRESSION`은 Supplementary Tables 1·2의 정규화 전 점수다(baseline은
4시간 튜닝, TabPFN은 기본 설정). 평균만 옮기고 표준편차는 뺐다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

PAPER_METHODS = (
    "AutoGluon",
    "CatBoost",
    "KNN",
    "LightGBM",
    "RF",
    "Linear",
    "MLP",
    "SVM",
    "XGB",
    "TabPFN (PHE)",
    "TabPFN",
)

# (이름, OpenML task id, dataset id, PAPER_METHODS 순서의 점수)
PAPER_CLASSIFICATION = (
    (
        "pc4",
        359958,
        1049,
        (0.9534, 0.9511, 0.8439, 0.9429, 0.9453, 0.9065, 0.9234, 0.9053, 0.9493, 0.9577, 0.9569),
    ),
    (
        "kc1",
        359962,
        1067,
        (0.8418, 0.8255, 0.7619, 0.8079, 0.8316, 0.7988, 0.8054, 0.7985, 0.8274, 0.8519, 0.8477),
    ),
    (
        "mfeatfactors",
        359961,
        12,
        (0.9996, 0.9989, 0.9969, 0.9991, 0.9986, 0.9985, 0.9989, 0.9995, 0.9991, 0.9997, 0.9998),
    ),
    (
        "bloodtransfusionservicecenter",
        359955,
        1464,
        (0.759, 0.7565, 0.7292, 0.744, 0.7346, 0.7592, 0.679, 0.7364, 0.7528, 0.7599, 0.7575),
    ),
    (
        "firstordertheoremproving",
        359969,
        1475,
        (0.8607, 0.8519, 0.7942, 0.8525, 0.8519, 0.7097, 0.7781, 0.7607, 0.8571, 0.8618, 0.8587),
    ),
    (
        "ozonelevel8hr",
        190137,
        1487,
        (0.9317, 0.9293, 0.8691, 0.9217, 0.912, 0.9152, 0.9176, 0.9212, 0.9321, 0.9343, 0.9271),
    ),
    (
        "phoneme",
        168350,
        1489,
        (0.9733, 0.964, 0.9319, 0.9599, 0.9631, 0.8136, 0.8913, 0.9127, 0.9667, 0.9734, 0.9729),
    ),
    (
        "qsarbiodeg",
        359956,
        1494,
        (0.9421, 0.9283, 0.9056, 0.9308, 0.9283, 0.9247, 0.927, 0.9225, 0.9324, 0.9401, 0.9361),
    ),
    (
        "yeast",
        2073,
        181,
        (0.8638, 0.8796, 0.8441, 0.8514, 0.8866, 0.8616, 0.8651, 0.8673, 0.8762, 0.8822, 0.8819),
    ),
    (
        "eucalyptus",
        359954,
        188,
        (0.9328, 0.9159, 0.8439, 0.913, 0.9142, 0.9138, 0.9095, 0.9067, 0.9049, 0.9361, 0.9341),
    ),
    (
        "cmc",
        359959,
        23,
        (0.7449, 0.748, 0.6377, 0.7464, 0.745, 0.7026, 0.697, 0.685, 0.7391, 0.7552, 0.7568),
    ),
    (
        "creditg",
        168757,
        31,
        (0.7968, 0.7972, 0.7431, 0.7873, 0.7998, 0.7879, 0.6698, 0.7855, 0.7962, 0.7966, 0.8029),
    ),
    (
        "krvskp",
        359965,
        3,
        (0.9998, 0.9997, 0.99, 0.9997, 0.9987, 0.9955, 0.9996, 0.9999, 0.9995, 0.9995, 0.9997),
    ),
    (
        "winequalitywhite",
        359974,
        40498,
        (0.8629, 0.8613, 0.7094, 0.8518, 0.8468, 0.7226, 0.7508, 0.8007, 0.8674, 0.9035, 0.904),
    ),
    (
        "dna",
        359964,
        40670,
        (0.9956, 0.9952, 0.9642, 0.9953, 0.9935, 0.9945, 0.9915, 0.9942, 0.9952, 0.9956, 0.9954),
    ),
    (
        "churn",
        359968,
        40701,
        (0.9327, 0.9224, 0.8479, 0.9198, 0.9166, 0.8621, 0.9129, 0.9013, 0.9203, 0.921, 0.9298),
    ),
    (
        "Satellite",
        359975,
        40900,
        (0.9961, 0.9923, 0.9275, 0.9849, 0.9828, 0.9922, 0.9944, 0.9917, 0.9872, 0.9977, 0.9963),
    ),
    ("car", 359960, 40975, (1.0, 1.0, 0.9927, 1.0, 0.9993, 0.9922, 0.9998, 1.0, 1.0, 1.0, 1.0)),
    (
        "Australian",
        146818,
        40981,
        (0.9428, 0.9406, 0.9216, 0.9413, 0.9389, 0.9309, 0.9236, 0.9299, 0.9425, 0.9456, 0.9433),
    ),
    (
        "steelplatesfault",
        168784,
        40982,
        (0.9716, 0.9694, 0.9245, 0.9668, 0.964, 0.9326, 0.9433, 0.9502, 0.9691, 0.9788, 0.9785),
    ),
    (
        "wilt",
        146820,
        40983,
        (0.9955, 0.993, 0.8539, 0.9922, 0.9877, 0.9751, 0.941, 0.9937, 0.9909, 0.9947, 0.9948),
    ),
    (
        "segment",
        359963,
        40984,
        (0.9968, 0.9961, 0.9867, 0.9956, 0.9956, 0.9853, 0.9851, 0.9916, 0.9963, 0.9972, 0.9974),
    ),
    (
        "jasmine",
        168911,
        41143,
        (0.8873, 0.8825, 0.8519, 0.8694, 0.8821, 0.8496, 0.8355, 0.8584, 0.8855, 0.8913, 0.8879),
    ),
    (
        "madeline",
        190392,
        41144,
        (0.9478, 0.9382, 0.6981, 0.9304, 0.8721, 0.6454, 0.7127, 0.7015, 0.9433, 0.975, 0.9748),
    ),
    (
        "philippine",
        190410,
        41145,
        (0.8799, 0.8681, 0.7735, 0.8678, 0.8441, 0.8018, 0.8025, 0.7973, 0.8747, 0.9751, 0.9724),
    ),
    (
        "sylvine",
        359972,
        41146,
        (0.9926, 0.9885, 0.8931, 0.9885, 0.9822, 0.966, 0.9745, 0.9731, 0.9882, 0.9972, 0.9964),
    ),
    (
        "ada",
        190411,
        41156,
        (0.922, 0.9199, 0.8721, 0.919, 0.9142, 0.9065, 0.8956, 0.8994, 0.923, 0.9182, 0.9172),
    ),
    (
        "GesturePhaseSegmentationProcessed",
        359970,
        4538,
        (0.9448, 0.92, 0.838, 0.9168, 0.8978, 0.7617, 0.7825, 0.7941, 0.921, 0.9493, 0.9437),
    ),
    (
        "vehicle",
        190146,
        54,
        (0.9756, 0.9376, 0.9031, 0.9322, 0.9315, 0.9486, 0.934, 0.9664, 0.94, 0.9738, 0.9732),
    ),
)

# (이름, OpenML task id, dataset id, PAPER_METHODS 순서의 점수)
PAPER_REGRESSION = (
    (
        "Moneyball",
        167210,
        41021,
        (
            -21.0201,
            -21.8662,
            -43.7007,
            -22.2264,
            -24.447,
            -21.1813,
            -21.5401,
            -22.4235,
            -22.4115,
            -20.7921,
            -20.9331,
        ),
    ),
    (
        "yprop_4_1",
        359940,
        416,
        (
            -0.0277,
            -0.028,
            -0.0287,
            -0.0281,
            -0.0278,
            -0.0286,
            -0.0328,
            -0.1389,
            -0.0278,
            -0.028,
            -0.0281,
        ),
    ),
    (
        "SAT11HANDruntimeregression",
        359948,
        41980,
        (
            -867.5137,
            -1087.0711,
            -1441.9111,
            -929.8491,
            -1103.3545,
            -1572.6944,
            -1140.7448,
            -1834.0961,
            -916.4924,
            -1000.6162,
            -1032.417,
        ),
    ),
    (
        "topo_2_1",
        359939,
        422,
        (
            -0.0281,
            -0.0283,
            -0.0293,
            -0.0283,
            -0.0285,
            -0.0286,
            -0.0296,
            -0.0321,
            -0.0282,
            -0.0282,
            -0.0284,
        ),
    ),
    (
        "house_prices_nominal",
        359951,
        42563,
        (
            -25931.8375,
            -24780.5884,
            -41721.4321,
            -26001.7092,
            -29216.3685,
            -33163.3526,
            -35394.7936,
            -72992.184,
            -36408.5046,
            -21826.569,
            -23394.6543,
        ),
    ),
    (
        "Mercedes_Benz_Greener_Manufacturing",
        233215,
        42570,
        (
            -8.2764,
            -8.271,
            -9.2702,
            -8.3343,
            -8.4401,
            -8.5241,
            -8.618,
            -8.657,
            -11.9073,
            -8.2135,
            -8.2414,
        ),
    ),
    (
        "abalone",
        359944,
        42726,
        (
            -2.0515,
            -2.1367,
            -2.1963,
            -2.1284,
            -2.1319,
            -2.2122,
            -2.193,
            -2.1295,
            -2.1117,
            -2.033,
            -2.034,
        ),
    ),
    (
        "colleges",
        359942,
        42727,
        (
            -0.1354,
            -0.1378,
            -0.1642,
            -0.139,
            -0.1453,
            -0.1605,
            -0.1581,
            -0.1517,
            -0.1449,
            -0.1357,
            -0.1359,
        ),
    ),
    (
        "us_crime",
        359945,
        42730,
        (
            -0.1301,
            -0.1304,
            -0.1459,
            -0.1305,
            -0.1346,
            -0.136,
            -0.1621,
            -0.1339,
            -0.1302,
            -0.1253,
            -0.1265,
        ),
    ),
    (
        "MIP2016regression",
        360945,
        43071,
        (
            -20560.6384,
            -21449.9103,
            -25957.1756,
            -20583.7643,
            -22929.1425,
            -25180.3342,
            -24861.1096,
            -31791.5722,
            -20838.3065,
            -19738.0161,
            -20277.5602,
        ),
    ),
    (
        "airfoil_self_noise",
        361235,
        44957,
        (
            -1.028,
            -1.0953,
            -2.1344,
            -1.3636,
            -1.6729,
            -4.8235,
            -4.4687,
            -3.2005,
            -1.205,
            -0.8748,
            -0.8598,
        ),
    ),
    (
        "auction_verification",
        361236,
        44958,
        (
            -241.2112,
            -511.0889,
            -3383.0746,
            -398.1578,
            -751.3872,
            -6185.0765,
            -2902.3171,
            -5465.7424,
            -341.1392,
            -377.8745,
            -397.0643,
        ),
    ),
    (
        "concrete_compressive_strength",
        361237,
        44959,
        (
            -3.6902,
            -3.689,
            -9.0353,
            -3.9162,
            -4.846,
            -10.4353,
            -7.9273,
            -6.2532,
            -3.8371,
            -3.6562,
            -3.6655,
        ),
    ),
    (
        "energy_efficiency",
        361617,
        44960,
        (
            -0.2606,
            -0.2617,
            -2.6296,
            -0.3143,
            -0.4585,
            -2.9429,
            -2.9727,
            -1.0502,
            -0.2784,
            -0.3722,
            -0.3563,
        ),
    ),
    (
        "geographical_origin_of_music",
        361243,
        44965,
        (
            -14.8767,
            -15.2125,
            -16.068,
            -15.5394,
            -15.8007,
            -16.6821,
            -17.6265,
            -15.7064,
            -15.2186,
            -14.3053,
            -14.4734,
        ),
    ),
    (
        "student_performance_por",
        361619,
        44967,
        (
            -2.6533,
            -2.6968,
            -2.9864,
            -2.6883,
            -2.6743,
            -2.7538,
            -2.7606,
            -2.73,
            -2.6564,
            -2.6362,
            -2.6372,
        ),
    ),
    (
        "QSAR_fish_toxicity",
        361621,
        44970,
        (
            -0.8706,
            -0.8736,
            -0.8898,
            -0.902,
            -0.8637,
            -0.9533,
            -0.9442,
            -0.8891,
            -0.8692,
            -0.8476,
            -0.856,
        ),
    ),
    (
        "grid_stability",
        361251,
        44973,
        (
            -0.0041,
            -0.0064,
            -0.0165,
            -0.0072,
            -0.0117,
            -0.022,
            -0.0169,
            -0.0369,
            -0.0072,
            -0.0047,
            -0.0047,
        ),
    ),
    (
        "cpu_activity",
        361256,
        44978,
        (
            -2.0489,
            -2.1149,
            -2.8326,
            -2.2425,
            -2.4535,
            -9.6325,
            -4.4565,
            -2.56,
            -2.1289,
            -2.0097,
            -2.0452,
        ),
    ),
    (
        "kin8nm",
        361258,
        44980,
        (
            -0.0626,
            -0.0901,
            -0.1152,
            -0.106,
            -0.1391,
            -0.2019,
            -0.1066,
            -0.0772,
            -0.105,
            -0.066,
            -0.0661,
        ),
    ),
    (
        "pumadyn32nh",
        361259,
        44981,
        (
            -0.0209,
            -0.0212,
            -0.0338,
            -0.0217,
            -0.0216,
            -0.033,
            -0.0344,
            -0.0348,
            -0.0215,
            -0.0206,
            -0.0207,
        ),
    ),
    (
        "cars",
        361622,
        44994,
        (
            -2080.5064,
            -2069.9186,
            -2534.9766,
            -2144.4847,
            -2211.9957,
            -2945.7366,
            -2168.9747,
            -5496.2717,
            -2115.3212,
            -1981.8505,
            -2023.9633,
        ),
    ),
    (
        "tecator",
        359934,
        505,
        (
            -0.6825,
            -1.3546,
            -4.7739,
            -1.7078,
            -1.394,
            -0.8571,
            -11.0025,
            -1.3195,
            -0.9707,
            -0.4612,
            -0.4453,
        ),
    ),
    (
        "space_ga",
        359933,
        507,
        (
            -0.0932,
            -0.1041,
            -0.1216,
            -0.103,
            -0.1154,
            -0.1286,
            -0.1438,
            -0.1032,
            -0.1066,
            -0.0915,
            -0.0913,
        ),
    ),
    (
        "boston",
        359950,
        531,
        (
            -2.7009,
            -2.8746,
            -4.5966,
            -3.1347,
            -3.2505,
            -4.7527,
            -4.9578,
            -3.499,
            -2.9128,
            -2.5158,
            -2.58,
        ),
    ),
    (
        "socmob",
        359932,
        541,
        (
            -11.8806,
            -15.1392,
            -17.4632,
            -14.5292,
            -18.5558,
            -24.882,
            -20.4125,
            -21.6415,
            -12.5944,
            -11.9149,
            -10.066,
        ),
    ),
    (
        "sensory",
        359931,
        546,
        (
            -0.6833,
            -0.7324,
            -0.7476,
            -0.6854,
            -0.6955,
            -0.7644,
            -0.7576,
            -0.7103,
            -0.6871,
            -0.6832,
            -0.6839,
        ),
    ),
    (
        "quake",
        359930,
        550,
        (
            -0.1882,
            -0.1884,
            -0.193,
            -0.1895,
            -0.1885,
            -0.1888,
            -0.19,
            -0.1959,
            -0.1885,
            -0.1889,
            -0.1902,
        ),
    ),
)


def paper_scores(task: str, method: str = "TabPFN") -> dict[int, float]:
    """논문 표의 점수 {OpenML task id: 점수}."""
    table = PAPER_CLASSIFICATION if task == "classification" else PAPER_REGRESSION
    column = PAPER_METHODS.index(method)
    return {task_id: scores[column] for _, task_id, _, scores in table}


def task_names(task: str) -> dict[int, str]:
    table = PAPER_CLASSIFICATION if task == "classification" else PAPER_REGRESSION
    return {task_id: name for name, task_id, _, _ in table}


@dataclass
class OpenMLTask:
    task_id: int
    name: str
    X: pd.DataFrame
    y: pd.Series
    categorical_indices: list[int]
    task_type: str  # "classification" | "regression"
    _task: object

    def folds(self, n_folds: int = 10) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for fold in range(n_folds):
            train_ix, test_ix = self._task.get_train_test_split_indices(
                fold=fold, repeat=0, sample=0
            )
            yield fold, train_ix, test_ix


def load_openml_task(task_id: int) -> OpenMLTask:
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    task_type = (
        "regression"
        if task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION
        else "classification"
    )
    cat_ix = [i for i, is_cat in enumerate(categorical) if is_cat]
    return OpenMLTask(task_id, dataset.name, X, y, cat_ix, task_type, task)


def score_fold(
    task_type: str, y_true: np.ndarray, prediction: np.ndarray, classes: np.ndarray | None = None
) -> float:
    """분류: ROC AUC (이진은 양성 클래스 확률, 다중은 one-vs-rest 평균). 회귀: negative RMSE."""
    if task_type == "regression":
        return -float(mean_squared_error(y_true, prediction) ** 0.5)
    # 학습 fold에 없던 클래스의 테스트 행은 모델이 예측할 수 없으므로 뺀다
    seen = np.isin(y_true, classes)
    y_true, prediction = y_true[seen], prediction[seen]
    if len(np.unique(y_true)) < 2:
        return float("nan")  # 클래스가 하나뿐이면 ROC AUC를 정의할 수 없다
    if prediction.shape[1] == 2:
        return float(roc_auc_score(y_true == classes[1], prediction[:, 1]))
    # 테스트 fold에 없는 클래스는 빼고 확률을 다시 정규화한다
    present = np.isin(classes, np.unique(y_true))
    proba = prediction[:, present]
    proba = proba / proba.sum(axis=1, keepdims=True)
    if present.sum() == 2:
        return float(roc_auc_score(y_true == classes[present][1], proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes[present]))


def evaluate_task(
    task: OpenMLTask,
    make_estimator: Callable[[OpenMLTask], object],
    n_folds: int = 10,
) -> list[dict]:
    rows = []
    for fold, train_ix, test_ix in task.folds(n_folds):
        estimator = make_estimator(task)
        x_train, x_test = task.X.iloc[train_ix], task.X.iloc[test_ix]
        y_train, y_test = task.y.iloc[train_ix].to_numpy(), task.y.iloc[test_ix].to_numpy()
        estimator.fit(x_train, y_train)
        if task.task_type == "classification":
            prediction = estimator.predict_proba(x_test)
            score = score_fold("classification", y_test, prediction, estimator.classes_)
        else:
            prediction = estimator.predict(x_test)
            score = score_fold("regression", y_test.astype(float), prediction)
        rows.append({"task_id": task.task_id, "dataset": task.name, "fold": fold, "score": score})
    return rows
