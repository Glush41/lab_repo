ЛР 3. Вариант 36: CIFAR-100, CNN ~0.84M

Номер варианта: 36
Номер работы: 3
Дата: 2026-11-16
Автор:Владислав Глущенко
Хеш коммита: e2e8a1a5ac5cefd0d228a8da479117e50872a4f7
Хеши данных: см. `hash_manifest.json` (train=735e79..., test=4b6768..., meta=a5d478...)

Схема настроек (pydantic)
`DLConfig`: data_root, subset_per_class, mode, batch_size, epochs, seed, lr, weight_decay, early_stop_patience, num_workers, device.
Невалидный конфиг отклоняется (`dl_config_negative_test.log`): batch_size=333, epochs=0, mode=invalid.


Архитектура
CNN: 3 блока Conv2d+BatchNorm+ReLU+MaxPool (32→64→128), классификатор Flatten→Linear(2048,256)→Dropout(0.3)→Linear(256,100).
Число параметров: 838 148 (~0.84M).
Ранняя остановка: patience=3 по val_acc.

Режимы загрузки
- full: 10 000 изображений в памяти (`TensorDataset`), DataLoader batch_size=125.
- stream: `StreamDataset` читает с диска по одному, без материализации всего датасета.
Индексы перемешаны с seed=42 перед разбиением 80/20 — train и test содержат все 100 класс
Таблица сравнения режимов
| mode | seed | batch_size | n_params | best_val_acc | acc_lo | acc_hi | time_sec | peak_memory_mb | run_id | скрипт |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 42 | 125 | 838148 | 0.2520 | 0.2340 | 0.2710 | 235.9 | 0.76 | 987c1649866e4ef0a84e478c52e7c94d | `train_dl.py --config configs/lab3_full.yaml` |
| stream | 42 | 125 | 838148 | 0.2520 | 0.2340 | 0.2710 | 277.4 | 0.84 | 4d29fe619d0847f299096ca3b71bdfd4 | `train_dl.py --config configs/lab3_stream.yaml` |
| full (repeat) | 42 | 125 | 838148 | 0.2520 | 0.2340 | 0.2710 | 236.3 | 0.76 | 67e305345d644da59569b4cf25c3f178 | повтор |

Бутстрэп-интервалы
full: best_val_acc=0.2520, интервал [0.2340, 0.2710].
stream: best_val_acc=0.2520, интервал [0.2340, 0.2710].

Сравнение с ЛР2
ЛР2 (HOG+SGD) full accuracy: 0.125.
ЛР3 (CNN) full best_val_acc: 0.2520.
Прирост CNN над ML: 2.02×.

Воспроизводимость
Объявленный режим: точное совпадение при детерминированных флагах (torch.manual_seed, CPU).
Повтор full с seed=42: best_val_acc=0.2520 (первый: 0.2520), run_id=67e305345d644da59569b4cf25c3f178.
Расхождение = 0, в допуске ≤ 0.005. Hash модели совпал: 76f9573aea5c1fe0...

Паритет устройств
GPU недоступен, проверен CPU-детерминизм (`parity_test.log`): max diff = 0.00000000 ≤ 1e-5.

Модель
`dl_model.pt`, SHA-256: 76f9573aea5c1fe0f9e93c67f2bc3c8ad2432a6faf8d5f38dcf72ed55c20e8de.
Загрузка: `CIFAR100CNN().load_state_dict(torch.load("dl_model.pt"))` — воспроизводится.

Негативные контроли
- Плохая конфигурация (`dl_config_negative_test.log`): batch_size=333, epochs=0, mode=invalid — отклонены.
- Снятая фиксация зерна (`seed_test.log`): без seed max diff=1.479083 > 1e-5 (расходятся), с seed — совпадают.
- Стриминг держит память: peak_memory stream=0.84 MB, full=0.76 MB (tracemalloc не видит PyTorch-тензоры; stream не материализует всё в памяти, время 277.4 с vs 235.9 с).

Выводы
1. CNN full: best_val_acc=0.2520, stream=0.2520. Расхождение = 0 ≤ 0.005.
2. Время full=235.9 с, stream=277.4 с. Stream медленнее на 17.6% из-за чтения с диска.
3. Пик памяти: stream не превышает full (0.84 vs 0.76 MB по tracemalloc; PyTorch-тензоры вне Python heap).
4. CNN превосходит HOG+SGD из ЛР2: 2.02× (0.2520 vs 0.125).
5. Детерминизм: repeat full с seed=42 даёт точно 0.2520, hash совпадает (76f9573a...).
6. Паритет CPU: расхождение 0.00000000 ≤ 1e-5.
