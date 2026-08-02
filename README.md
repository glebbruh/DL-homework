# Voice Anti-Spoofing

Выполнил: **Полетаев Глеб**

___

Проект выполнен в рамках домашнего задания по курсу **Deep Learning Research**.

В работе решается задача обнаружения поддельной речи (Voice Anti-Spoofing) на разделе **Logical Access (LA)** датасета **ASVspoof 2019**. В качестве модели используется **Light Convolutional Neural Network (LCNN)**, реализованная в соответствии с требованиями задания. Проект основан на **PyTorch Project Template**, использует **Hydra** для управления конфигурациями и **Weights & Biases (W&B)** для логирования экспериментов.

## Датасет

Для обучения и оценки используется датасет **ASVspoof 2019 LA**.

Необходимые части датасета:

- ASVspoof2019_LA_train
- ASVspoof2019_LA_dev
- ASVspoof2019_LA_eval
- ASVspoof2019_LA_cm_protocols


## Модель

В проекте реализована архитектура **LCNN** с блоками **Max-Feature-Map (MFM)**.

В качестве входных признаков используется **Log-Power Spectrogram**, вычисляемая с помощью STFT.

Обучение выполняется как задача бинарной классификации:

- **bonafide** — настоящая речь;
- **spoof** — синтезированная или преобразованная речь.

## Структура проекта

```text
.
├── grading/
├── src/
│   ├── configs/
│   ├── datasets/
│   ├── logger/
│   ├── loss/
│   ├── metrics/
│   ├── model/
│   ├── trainer/
│   ├── transforms/
│   └── utils/
├── tests/
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## Установка

Создание виртуального окружения:

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Для логирования экспериментов необходимо авторизоваться в Weights & Biases:

```bash
wandb login
```

При запуске в Kaggle API-ключ рекомендуется сохранить в Secrets под именем:

```text
WANDB_API_KEY
```

## Обучение

Основной конфигурационный файл расположен в:

```text
src/configs/lcnn_asvspoof.yaml
```

Запуск обучения:

```bash
python train.py -cn=lcnn_asvspoof
```

Все параметры эксперимента задаются через конфигурационные файлы Hydra.


## Метрика

Основная метрика качества — **Equal Error Rate (EER)**.

Итоговые предсказания проверяются с помощью предоставленного скрипта `grading.py`.

## Результаты

| Метрика | Значение |
|---------|---------:|
| EER | **9.8687%** |


## Логи обучения

Все эксперименты логируются в **Weights & Biases**.


**W&B Report:**

```
https://api.wandb.ai/links/glebbrr/ygazr4su
```

## Использованные материалы

При реализации проекта использовались материалы домашнего задания и статьи, указанные в задании:

- *Light CNN for Deep Face Representation with Noisy Labels*;
- *STC Antispoofing Systems for the ASVspoof2019 Challenge*;
- *A Comparative Study on Recent Neural Spoofing Countermeasures*.