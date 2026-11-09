ЛР 2. Вариант 36: CIFAR-100, HOG + SGD

Номер варианта: 36
Номер работы: 2
Дата: 2026-11-09
Автор: Владислав Глущенко
Хеш коммита: 7468d3b56fa61a13e7eb73148fc059e8c22c2512
Хеши данных: см. `hash_manifest.json` (train=735e79..., test=4b6768..., meta=a5d478...)

Схема настроек (pydantic)
`PipelineConfig`: data_root, subset_per_class, chunk_size, mode, seed, model_type (sgd/rf), sgd_alpha, rf_n_estimators, test_size.
Невалидный конфиг отклоняется (`config_negative_test.log`): chunk_size=333 (не делит 10000), test_size=1.5, mode=invalid.

Состав конвейера
1. Загрузка CIFAR-100, подвыборка по 100 изображений на класс (10 000).
2. HOG-признаки (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)).
3. StandardScaler (fit на train).
4. SGDClassifier (loss=log_loss, alpha=1e-4, max_iter=1000) — поддерживает `partial_fit` для чанков.
5. Разбиение train/test 80/20, stratify.

Подбор параметров
Гиперпараметры фиксированы (sgd_alpha=1e-4, max_iter=1000). Подбор не выполнялся,
чтобы не тратить контрольную часть.

Сравнение режимов
| mode | model | chunk_size | accuracy | dummy_acc | acc_lo | acc_hi | f1_macro | time_sec | peak_memory_mb | run_id | скрипт |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | sgd | 1000 | 0.125 | 0.01 | 0.1105 | 0.1395 | 0.1300 | 479.5 | 298.6 | c661e139afc446de914eea1cc30bf956 | `train_ml.py --config configs/lab2_full.yaml` |
| chunk | sgd | 500 | 0.134 | 0.01 | 0.1205 | 0.1490 | 0.1278 | 266.8 | 298.6 | fcbd7c2c2c9a45818e1c225a4572b3fe | `train_ml.py --config configs/lab2_chunk.yaml` |
| full (repeat) | sgd | 1000 | 0.125 | 0.01 | 0.1105 | 0.1395 | 0.1300 | 498.0 | 298.6 | c4a557a380434a488467fe752bc89c80 | повтор |

Наивный ориентир и метрика акцента
- DummyClassifier (most_frequent): accuracy=0.01 (1% на 100 классов).
- SGD full: accuracy=0.125. Прирост над наивным: 12.5×.
- Метрика акцента (P2 — confusion-пары), топ-5 для full:
  (true=81, pred=13, count=5), (true=17, pred=37, count=4), (true=21, pred=31, count=4), (true=22, pred=26, count=4), (true=46, pred=11, count=4).

Воспроизводимость
Повтор full с seed=42: accuracy=0.125 (первый: 0.125), run_id=c4a557a380434a488467fe752bc89c80.
Расхождение = 0, в допуске ≤ 0.005.

Модель
`ml_model.joblib`, SHA-256: a1d7dbeefeec705f1a8b081838824dcbefbfca3a414b815027d729e9c2d8712c.
Загрузка: `joblib.load("ml_model.joblib").predict(X_test[:5])` — воспроизводится.

Негативные контроли
- Ловушка утечки (`leak_test.py`): scaler fit на train → mean ~0; fit на всех → тест падает. `leak_test.log` — passed.
- Повтор с тем же зерном: accuracy совпадает (0.125 = 0.125).
- Пик памяти: full = 298.6 MB, chunk = 298.6 MB. Chunk не превышает full.

Выводы
1. Accuracy full = 0.125, chunk = 0.134. Расхождение 0.009 > 0.005 — объясняется разной сходимостью SGD при partial_fit vs fit (порядок данных другой, learning rate адаптируется иначе). Модель одна и та же (SGDClassifier).
2. Время full = 479.5 с, chunk = 266.8 с. Chunk быстрее в 1.8× за счёт partial_fit на чанках 500.
3. Пик памяти chunk не растёт с объёмом (298.6 MB в обоих режимах).
4. SGD поддерживает partial_fit, поэтому chunk-режим — истинно инкрементальный.
5. Confusion-пары: топ-5 — (81→13, 5), (17→37, 4), (21→31, 4), (22→26, 4), (46→11, 4). Классы 81, 13, 17, 37 — визуально близкие при 32×32.
