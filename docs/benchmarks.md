# Benchmarks

## Overview

Бенчмарки используют [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) и запускаются через
BitNet inference server (HTTP API). Адаптер `BitNetLM` реализует интерфейс `lm_eval.api.model.LM` для двух режимов
BitNet (`chat` / `completion`).

## Benchmarks Module

```
benchmarks/
├── run_all.py            Запуск всех бенчмарков
├── run_mmlu_pro.py       MMLU-Pro (14 subjects)
├── run_ifeval.py         IFEval (instruction following)
├── run_medqa.py          MedQA (4 options)
├── run_meddialog.py      MedDialog (dialogue, qsumm, perplexity)
├── test_stub.py          Smoke-тест с заглушкой
├── lm_eval_adapter.py    BitNetLM — адаптер для lm-eval-harness
├── stub_lm.py            StubLM — заглушка для отладки (ответ = gold)
└── _utils.py             Сохранение результатов (JSON + summary table)
```

## BitNetLM — адаптер

`BitNetLM` в `benchmarks/lm_eval_adapter.py`:

- Реализует `generate_until()` и `loglikelihood()` (приближённо, через генерацию).
- `loglikelihood_rolling()` не поддерживается — BitNet HTTP API не отдаёт per-token logprobs.
- В `generate_until()` контекст отправляется как `user`-сообщение (chat) или как `prompt` (completion).
- Параметры генерации (`max_tokens`, `temperature`, `stop`, `do_sample`) берутся из `gen_kwargs` от lm-eval.

## StubLM — заглушка

`StubLM` в `benchmarks/stub_lm.py`:

- Всегда возвращает правильный ответ (gold label).
- Используется для проверки корректности пайплайна без реального LLM.
- Поддерживает MMLU-Pro, MedQA и IFEval.

## Установка

```bash
pip install -r requirements-bench.txt
```

Устанавливает всё из `requirements.txt` (RAG API) плюс lm-eval-harness, датасеты и метрики
(bert-score, rouge-score, BLEURT, evaluate).

Полный frozen-слепок окружения — `requirements_full.txt`.

## Запуск

### Все бенчмарки

```bash
python -m benchmarks.run_all --model "bitnet-2B"
```

### Отдельный бенчмарк

```bash
python -m benchmarks.run_mmlu_pro --model "bitnet-2B" --limit 10

python -m benchmarks.run_mmlu_pro \
  --model "bitnet-2B" \
  --subjects mmlu_pro_health mmlu_pro_math \
  --limit 5
```

### Smoke-тест с StubLM

```bash
python -m benchmarks.test_stub --limit 3
```

## Результаты

Результаты сохраняются в `logs/benchmarks/<benchmark_name>/`:

```
logs/benchmarks/
└── mmlu_pro/
    ├── results_mmlu_pro_health_bitnet-2B_20260626_174957.json
    ├── results_mmlu_pro_health_bitnet-2B_20260626_174957_summary.txt
    ├── results_mmlu_pro_health_bitnet-2B_latest.json          # symlink/snapshot
    └── results_mmlu_pro_health_bitnet-2B_latest_summary.txt
```

Каждый запуск создаёт:

- `*.json` — полные результаты (config, samples, metrics).
- `*_summary.txt` — таблица метрик (exact_match и др.).

Файлы `_latest.*` перезаписываются при каждом запуске для быстрого доступа.
