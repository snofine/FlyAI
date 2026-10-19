# FlyBrain (FlyAI) — Drosophila Connectome Simulation

> ⚡ **Architected, modeled, and accelerated using OpenAI Codex**

A computational model of a virtual *Drosophila melanogaster*, with neural network connectivity structured from the official aggregated **MaleCNS v1.0** connectome. 

### 🧠 Core Architecture
* **Connectome Integration**: Implements neural wiring inspired by the MaleCNS v1.0 dataset.
* **Sensory-Motor Pipeline**: Strict sensory encoding invariants (`looming → sensory encoder → descending/motor circuits`).
* **Neuro-Behavioral Logic**: Biological reaction modeling without hardcoded motion shortcuts.
* **Modern Tooling**: Zero-config execution supported via `uv`.

---


Вычислительная модель виртуальной дрозофилы, структура связей которой может быть построена из официального агрегированного коннектома **MaleCNS v1.0**. Это не «точная копия мозга»: электрические параметры, рецепторы и сенсорная трансдукция в текущем MVP упрощены.

Главный инвариант проекта:

```text
looming → sensory encoder → sensory neurons → CSR connectome
        → descending/motor groups → temporal decoder → 2D body
```

Раздражитель нигде не вызывает `escape()` или движение напрямую.

## Быстрый запуск

Системный Python не требуется, если установлен `uv`:

```powershell
uv sync --extra ui --extra dev
uv run flybrain --demo
```

Встроенный demo-граф состоит из 8 искусственных узлов и нужен только для проверки полного контура без загрузки 1.1 ГБ. В интерфейсе он явно помечен `DEMO graph`.

Headless-проверка:

```powershell
uv run flybrain --headless --demo --seconds 5
uv run pytest
```

## MaleCNS v1.0

Скачать ровно три требуемых файла (примерно 1.2 ГБ):

```powershell
uv run flybrain-download
```

Можно сначала скачать только аннотации и нейромедиаторы:

```powershell
uv run flybrain-download --skip-weights
```

Конвертация в memory-mapped CSR (два потоковых прохода по Feather; весь edge-list не загружается в pandas):

```powershell
uv run flybrain-preprocess --minimum-weight 1 --normalization log1p --weight-scale 0.02 --nt-mode experimental
```

По умолчанию сохраняются строки с назначенным `superclass` (166 700 нейронов в официальных файлах v1.0); glia, orphan/unimportant и прочие неклассифицированные сегменты исключаются. Для диагностической сборки всех аннотированных тел есть `--include-unclassified`.

После этого обычный запуск автоматически использует `data/processed`:

```powershell
uv run flybrain
uv run flybrain --debug
uv run flybrain --headless --seconds 30
```

### Ручные раздражители

В интерактивном окне раздражители подаются клавишами и всегда проходят через реальные сенсорные группы MaleCNS:

```text
1 / 2  — левый / правый глазной сенсорный канал
3      — передние лапки
4      — средние лапки
5      — задние лапки
6      — механосенсорика головы
```

Из командной строки можно назначить время и длительность; `--stimulate` допускается повторять:

```powershell
uv run flybrain --headless --seconds 3 `
  --stimulate eye_left --stimulate leg_front `
  --stimulus-at 1.0 --stimulus-duration 250
```

Клик по мухе включает debug-панель. Body ID показываются за скользящее окно 100 мс, поэтому редкие spikes больше не теряются между графическими кадрами.

Исходные файлы ожидаются под короткими локальными именами:

```text
data/raw/annotations.feather
data/raw/neurotransmitters.feather
data/raw/connectome-weights.feather
```

Официальные источники: [MaleCNS project](https://male-cns.janelia.org/) и [download page](https://male-cns.janelia.org/download/). Данные MaleCNS распространяются по CC-BY; при публикации производных результатов сохраните атрибуцию датасета.

## Роли нейронов

`config/sensory_neurons.yaml` и `config/motor_neurons.yaml` поддерживают два способа выбора:

- точные реальные `body_ids` — предпочтительный вариант для воспроизводимого эксперимента;
- регулярные выражения по аннотациям — bootstrap для первого построения.

Preprocessor сохраняет выбранные плотные индексы в `data/processed/neuron_roles.json`. Перед научным экспериментом проверьте группы через neuPrint и зафиксируйте конкретные body ID в YAML.

## Инструменты

```powershell
# Метаданные, суммарные связи и top-20 партнёров
uv run flybrain-inspect 12781

# Сохранить/продолжить полное динамическое состояние
uv run flybrain --headless --seconds 5 --snapshot snapshots/run.npz
uv run flybrain --headless --seconds 5 --snapshot snapshots/run.npz

# Повторно вычислить эксперимент и сверить новый временной ряд с исходным
uv run flybrain --headless --replay logs/experiment_YYYYMMDD_HHMMSS
```

Каждый запуск пишет `manifest.json` (seed, конфигурация и параметры сборки коннектома) и `steps.parquet` (стимул, spikes, motor output и положение тела). Replay завершается кодом 0 только при численном совпадении нового ряда с исходным.

## Структура

```text
src/flybrain/
  brain/       CSR, LIF, event-driven propagation
  sensory/     binocular vision and looming encoder
  motor/       rolling-window decoder
  body/        2D fly dynamics
  world/       arena and approaching threat
  ui/          Pygame renderer and group activity panel
  tools/       download, preprocessing, neuron inspection
```

## Производительность и границы MVP

CSR-массивы открываются через `numpy.memmap`. На каждом шаге обходятся исходящие рёбра только спайкнувших нейронов. Preprocessing всё же требует место для исходного Feather и примерно 1–2 ГБ для массивов графа; итог зависит от фильтра веса.

Нейтромедиаторы сохраняются в metadata. Для интерактивного режима рекомендуется `--nt-mode experimental`: он предотвращает глобальную лавину all-positive сети, но таблица знаков в `config/neurotransmitters.yaml` остаётся приближением и не заявляется как биологически точная. Режим `--nt-mode neutral` оставлен для проверки распространения: все связи в нём положительные, поэтому на полном рекуррентном графе возможна насыщенная активность и низкая скорость.

Текущий MVP реализует looming. Обоняние, прикосновение, температура, пластичность и детальная нейромодуляция оставлены следующими этапами.
