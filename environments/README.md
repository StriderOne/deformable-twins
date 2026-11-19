# Окружения: Isaac Lab и урок PhysTwin → Newton

Для калибровки см. [отдельное окружение PhysTwin VI/MAP](README-vi.md).

Отдельный контейнер с RTX-окном, мешами Panda и Gaussian Splatting:
[Isaac Sim + PhysTwin](README-isaac-sim.md).

Для [учебного notebook](../tutorials/phystwin_to_newton.ipynb) используйте
[отдельный venv на хосте](#notebook-перенос-phystwin). Ниже также сохранены
команды Docker для полноценных сцен Isaac Lab.

Контейнер с Isaac Lab 3.0.0 и Newton, без полного редактора Isaac Sim.
Локальное окно сцены открывается через `newton_gl` и X11.
Базовый образ закреплён по digest в [Dockerfile](Dockerfile);
зависимости уже установлены в нём.

Требуются Linux x86_64, Docker с NVIDIA Container Toolkit, NVIDIA GPU,
локальная сессия X11 и `xauth`. Пользователь контейнера — `isaaclab`, UID/GID
`1000:1000`; приведённые команды рассчитаны на такой же UID/GID на хосте.
Установка Isaac и Conda на хосте не используется.

## Сборка

Выполняйте команды в Bash из корня этого репозитория.
Проверяйте свободное место до загрузки и после сборки. Сжатые слои базового
образа занимают около 5,47 GiB; для распаковки нужно дополнительное место.
При остатке менее 10 GiB остановитесь; команды не удаляют чужие образы и данные.

```bash
(
  set -euo pipefail
  df -h /
  test "$(df -B1 --output=avail / | tail -1)" -ge 10737418240
  docker build --tag deformable-twins/isaac-newton:3.0.0 - < environments/Dockerfile
  df -h /
  test "$(df -B1 --output=avail / | tail -1)" -ge 10737418240
)
```

Контекст сборки пустой: файлы проекта не отправляются в Docker.

## Общие параметры запуска

Выполните этот блок один раз в текущем терминале. Все кэши и технические
логи хранятся вне репозитория. Повторный запуск использует тот же кэш Warp.

```bash
isaac_cache="$HOME/.cache/deformable-twins/isaac-newton"
isaac_image=deformable-twins/isaac-newton:3.0.0
mkdir -p "$isaac_cache"/{warp,nvidia,compute,logs}
isaac_docker=(
  docker run --rm --gpus all --shm-size=1g
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
  --mount "type=bind,src=$isaac_cache/warp,dst=/home/isaaclab/.cache/warp"
  --mount "type=bind,src=$isaac_cache/nvidia,dst=/home/isaaclab/.cache/nvidia"
  --mount "type=bind,src=$isaac_cache/compute,dst=/home/isaaclab/.nv/ComputeCache"
  --mount "type=bind,src=$isaac_cache/logs,dst=/workspace/isaaclab/logs"
)
set -o pipefail
```

## Проверка GPU и запуск без окна

```bash
"${isaac_docker[@]}" "$isaac_image" nvidia-smi
"${isaac_docker[@]}" "$isaac_image" python -c '
from importlib import metadata
import torch
import warp as wp
for name in ("isaaclab", "newton", "warp-lang", "torch"):
    print(name, metadata.version(name))
assert torch.cuda.is_available(), "PyTorch CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
assert wp.get_device("cuda:0").is_cuda
print("CUDA allocation:", torch.ones(1, device="cuda").cpu())
'

"${isaac_docker[@]}" "$isaac_image" isaaclab random_agent \
  --task Isaac-Cartpole-Direct --help

"${isaac_docker[@]}" "$isaac_image" isaaclab random_agent \
  --task Isaac-Cartpole-Direct physics=newton_mjwarp \
  --num_envs 1 --viz none --max_steps 100 \
  2>&1 | tee "$isaac_cache/logs/headless.log"
```

Это проверка инфраструктуры со случайными действиями, без обучения политики.
Первый запуск компилирует GPU-ядра и может занять несколько минут.

## Графическое окно

Следующий блок запускается в отдельной оболочке, создаёт временный X11-cookie
с правами `600` и удаляет его при завершении. Значение cookie не выводится
в терминал. Общий доступ через `xhost +` не включается.

```bash
(
  set -euo pipefail
  test -n "${DISPLAY:-}"
  test -S /tmp/.X11-unix/X0
  isaac_xauth=$(mktemp "${XDG_RUNTIME_DIR:-/tmp}/isaac-newton-xauth.XXXXXX")
  chmod 600 "$isaac_xauth"
  trap 'rm -f -- "$isaac_xauth"' EXIT
  xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$isaac_xauth" nmerge -
  test -s "$isaac_xauth"

  "${isaac_docker[@]}" --name isaac-newton-gui \
    --env DISPLAY --env XAUTHORITY=/tmp/isaac-newton.xauth \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
    --mount "type=bind,src=$isaac_xauth,dst=/tmp/isaac-newton.xauth,readonly" \
    "$isaac_image" 2>&1 | tee "$isaac_cache/logs/gui.log"
)
```

Команда использует CMD из Dockerfile: одна среда Cartpole, Newton/MJWarp,
визуализатор `newton_gl`. Закройте окно для завершения контейнера.
Проверьте, что сцена движется и камера реагирует на мышь.
Для другого номера дисплея замените проверку сокета `X0` соответствующим номером.

## Franka и ткань

После блока общих параметров можно запустить штатный сценарный пример:
манипулятор пытается захватить и поднять ткань с опор. Newton использует
MJWarp для робота и VBD для ткани. Обученная политика и checkpoint не нужны.
Если предыдущий контейнер ещё работает, сначала остановите его.

```bash
(
  set -euo pipefail
  test -n "${DISPLAY:-}"
  test -S /tmp/.X11-unix/X0
  isaac_xauth=$(mktemp "${XDG_RUNTIME_DIR:-/tmp}/isaac-newton-xauth.XXXXXX")
  chmod 600 "$isaac_xauth"
  trap 'rm -f -- "$isaac_xauth"' EXIT
  xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$isaac_xauth" nmerge -
  test -s "$isaac_xauth"

  "${isaac_docker[@]}" --name isaac-newton-gui \
    --env DISPLAY --env XAUTHORITY=/tmp/isaac-newton.xauth \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
    --mount "type=bind,src=$isaac_xauth,dst=/tmp/isaac-newton.xauth,readonly" \
    "$isaac_image" python scripts/environments/state_machine/lift_franka_soft.py \
      --task Isaac-Lift-Cloth-Franka physics=newton_mjwarp_vbd_proxy \
      --device cuda:0 --num_envs 1 --num_steps 10000 --viz newton_gl \
    2>&1 | tee "$isaac_cache/logs/cloth.log"
)
```

Первый кадр может появиться через несколько минут: компилируются новые ядра.
Запуск ограничен 10000 шагами; для досрочной остановки используйте
`docker stop isaac-newton-gui`. В отличие от `random_agent`, этот upstream-скрипт
выполняет цикл фиксированной длины и не проверяет закрытие окна на каждом шаге.
Он использует curriculum гравитации и отключает часть условий завершения
эпизода. Это демонстрация взаимодействия, а не протокол оценки захвата
или физической точности.

## Notebook: перенос PhysTwin

Практикум: [tutorials/phystwin_to_newton.ipynb](../tutorials/phystwin_to_newton.ipynb).
Он запускается **на хосте в отдельном venv** `~/.venvs/phystwin-newton` с Python 3.12.
Здесь достаточно Newton и Warp; Isaac Lab и Docker для урока не требуются.
Задания, проверки, подсказки и интерактивные графики доступны прямо в notebook.

### Создание окружения

На этой машине окружение уже создано. Для повторения на другой машине выполните
из корня репозитория (нужен Python 3.12 с модулем venv):

```bash
python3.12 -m venv "$HOME/.venvs/phystwin-newton"
"$HOME/.venvs/phystwin-newton/bin/python" -m pip install \
  -r environments/requirements-notebook.txt
"$HOME/.venvs/phystwin-newton/bin/python" -m ipykernel install --user \
  --name phystwin-newton --display-name "Python (PhysTwin → Newton)"
```

Версии основных зависимостей закреплены в
[requirements-notebook.txt](requirements-notebook.txt). Окружение не наследует
пакеты из системного Python или Conda.

### Запуск в VS Code

Откройте notebook и через **Select Kernel** выберите
**Python (PhysTwin → Newton)**. Если список не обновился, перезагрузите окно VS Code.
Также можно выбрать интерпретатор напрямую:
`~/.venvs/phystwin-newton/bin/python`.

### Запуск в браузере

Из корня репозитория:

```bash
"$HOME/.venvs/phystwin-newton/bin/python" -m jupyterlab \
  --ip=127.0.0.1 --no-browser --ServerApp.root_dir="$PWD" \
  tutorials/phystwin_to_newton.ipynb
```

Откройте напечатанную сервером ссылку с токеном. Для остановки нажмите `Ctrl+C`
в терминале и подтвердите остановку сервера. Активация venv необязательна:
команды выше явно выбирают нужный Python. При желании активируйте его командой
`source "$HOME/.venvs/phystwin-newton/bin/activate"`.

Урок по умолчанию работает на CPU. Для GPU поменяйте `DEVICE` в начальной ячейке
на `cuda:0` и перезапустите ядро; нужен совместимый драйвер NVIDIA.
Графики внутри notebook не требуют X11 или отдельного Newton Viewer.
Кэш урока находится в `~/.cache/deformable-twins/phystwin-lesson/`.

Для своего экспорта задайте `REAL_EXPORT = Path('/absolute/path/to/export')`
с файлами `graph.npz` и `provenance.json` на хосте. Импорт не загружает исходные
pickle и не запускает реальный объект автоматически.

Проверено на хосте 2026-11-03: `pip check` проходит, notebook с решениями
выполняет 8/8 проверок на CPU и RTX 5060 через ядро `phystwin-newton`.
Исходный notebook с TODO выполняется без ошибок и показывает 0/8.
Локальный Jupyter отдаёт notebook через HTTP API с авторизацией.
Проверочные копии, полный список установленных пакетов и manifest находятся в
`~/.cache/deformable-twins/phystwin-lesson/host-validation/`.
Выбор ядра и управление виджетом через интерфейс VS Code вручную не проверялись.

## Shell и остановка

Пока графическое окно открыто, в другом терминале:

```bash
docker exec -it isaac-newton-gui bash
```

Для отдельного shell после выполнения блока общих параметров:

```bash
"${isaac_docker[@]}" -it "$isaac_image" bash
```

Для остановки окна из другого терминала:

```bash
docker stop isaac-newton-gui
```

Контейнер удаляется автоматически благодаря `--rm`; образ и кэш сохраняются.
Если образ больше не нужен, удалить только нашу метку можно командой
`docker image rm deformable-twins/isaac-newton:3.0.0`.

## Ограничения и проверка

GPU с 8 GiB VRAM используется для сцен с одной средой.
Масштабирование на множество сред и обучение не проверялись.
При ошибке драйвера Docker использует драйвер хоста: установка CUDA внутри
контейнера не обновляет его. При ошибке X11 проверьте `DISPLAY`, `XAUTHORITY`
и доступ к локальной графической сессии.

Локальная проверка 2026-10-29: Ubuntu с X11, RTX 5060 (8151 MiB),
драйвер 580.159.04. Образ собран, GPU доступен через `nvidia-smi`, PyTorch
и Warp; запуск без окна с `--max_steps 100` завершился с кодом 0.
Окно Newton Viewer открылось через X11, изображение Cartpole проверено
по снимку окна; движение сцены и управление камерой подтверждены пользователем.
Штатный запрос закрытия окна завершил процесс с кодом 0, контейнер автоматически
удалён, временный X11-cookie удалён. При повторной инициализации решателя использован кэш
(около 0,83 с против 14,45 с при первом запуске; это диагностика, не benchmark).

В образе `/workspace/isaaclab/VERSION` содержит `3.0.0`, версии пакетов:
`isaaclab==28.0.0`, `newton==1.6.1`, `warp-lang==1.17.0`,
`torch==2.11.0+cu130`. Версия Python-пакета Isaac Lab отличается от версии
релизного образа. После загрузки и сборки осталось около 20 GiB свободного места.

Дополнительно запущен сценарный пример `Isaac-Lift-Cloth-Franka` с одной средой:
сцена создана на GPU, окно с Franka и тканью отрисовано. Это проверка запуска
и визуализации; частота успешных захватов и точность динамики не измерялись.

Источники: [установка и контейнеры Isaac Lab 3.0](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html),
[команды и визуализаторы](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/quickstart.html).

## Свободный PhysTwin-объект в Newton

По умолчанию [`view_phystwin_gaussians.py`](../tutorials/view_phystwin_gaussians.py)
показывает свободный деформируемый объект, стол и **Franka Emika Panda**.
Внутренние силы вычисляет исходный kernel PhysTwin; динамику руки, частиц
и двусторонние контакты — `SolverFeatherstone` Newton. Контрольных точек объекта нет.
Прежняя сцена с шаром и `SolverSemiImplicit` доступна через `--sphere`.

```bash
~/.venvs/phystwin-newton/bin/python -m pip install -r environments/requirements-gaussian-viewer.txt
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py
```

Открыть **http://127.0.0.1:8080**. `--port 8082` запускает отдельный экземпляр.
По умолчанию запускается автоматический цикл: после оседания игрушки выбирается
случайная точка её верхней поверхности, робот подходит сверху с открытым
захватом, опускается, закрывает пальцы, поднимает руку на 30 см и сбрасывает
сцену. Следующий цикл выбирает новую точку. Красный маркер показывает выбранную
точку. Пауза останавливает и физику, и последовательность; «Сбросить» начинает
новую попытку. Без подключённых браузеров время не продвигается.

```bash
# Воспроизводимая последовательность случайных целей:
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --seed 0
```

[`grasp_cycle.py`](../src/lab/simulation/grasp_cycle.py) выбирает верхнюю физическую
частицу в каждой ячейке XY размером 1 см, затем случайную ячейку. Это поверхность
контактной модели, не Gaussian-оболочки. IK готовит последовательность целей
суставов для подхода, спуска и подъёма; физическое состояние руки не перезаписывается.
Пальцы удерживают объект только контактами: неудачный захват и выскальзывание
возможны. Если IK не находит допустимую позу, попытка записывается как неудачная,
после чего сцена сбрасывается. Параметры контакта и физика игрушки автоматически не подстраиваются.

В `events.jsonl` записываются seed, выбранная частица/позиция, фазы и цели,
изменение высоты частиц после подъёма, ошибки IK и автоматические сбросы.
Эти высотные метрики — диагностика демонстрации, не протокол оценки захвата.

Модель Panda скачивается в кэш Newton вне репозитория из MuJoCo Menagerie,
commit `da76818e269b82289eba39808e2fb91d679d6994`, лицензия Apache-2.0.
Это именно Panda, не FR3 из похожего каталога Newton assets.
[`panda.py`](../src/lab/simulation/panda.py) задаёт параметры PD-приводов,
ограничения усилия и скорости изменения целей. Суставы испытывают гравитацию
и ответные контактные силы, поэтому фактическая поза отличается от заданной.
Два пальца имеют независимые приводы с общей целью раскрытия; MJCF tendon
не воспроизводится. Самостолкновения робота отключены. Это демонстрационный
контроллер, а не воспроизведение аппаратного контроллера Franka.

```bash
# Предыдущая сцена: тянуть стрелки цели шара, отключить «Тянуть шар» для свободного движения.
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --sphere
```

Авторитетные демонстрационные параметры заданы в `DemoSettings` в
[`demo.py`](../src/lab/simulation/demo.py). Масса игрушки — допущение, а не
результат измерения; изменить её можно через `--object-mass` (килограммы).
Массы частиц, Y, границы Y и dashpot масштабируются одним коэффициентом.
Drag сохраняет размерность 1/с. Материалы контактов — новые некалиброванные
параметры Newton; коэффициенты столкновения с землёй PhysTwin не используются.

### Границы реализации

- В сцене остаются все 6 487 физических частиц и 52 778 внутренних пружин.
  30 управляющих точек и 388 управляющих пружин исключены.
- Физическое состояние находится только в `newton.Model` / `newton.State`.
  Исходные координаты повёрнуты в Z-up через `diag(1,-1,-1)` и подняты над столом;
  то же преобразование применяется к гауссианам.
- Контактная оболочка — сферы на частицах; она не совпадает точно с видимой
  поверхностью. Самостолкновения и контакты двух деформируемых объектов отключены.
- На каждом подшаге Newton вычисляет контакты с динамическими телами и плоскостью.
  Обработка земли PhysTwin не запускается. Предел скорости не используется
  как средство стабилизации; нефинитное состояние останавливает расчёт.
- Отрисовывается до 40 000 гауссиан с opacity > 0.1; `--max-gaussians` изменяет
  только детализацию рендера. Используется прежняя приближённая привязка по
  16 соседям и цвет SH нулевого порядка. Интерактивный режим поддерживает
  изотропные гауссианы. Время симуляции не привязано к реальному времени.
- Это forward-only инфраструктурная демонстрация. Не подтверждены физическая
  калибровка, устойчивость произвольных сцен, надёжный захват/подъём, обучение и sim-to-real.
  Физический робот не подключается; проверяется только взаимодействие в симуляции.

### Замена закона внутренних сил

[`internal_forces.py`](../src/lab/simulation/internal_forces.py) содержит
`InternalForceModel` Protocol, `SolverInternalForces` для шара и
`SolverArticulatedInternalForces` для робота. Оба используют одинаковые силы
PhysTwin, drag и контактный kernel частиц Newton.
Компонент реализует `accumulate(model, state, dt)` и `reset()`.
Первый метод **добавляет** силы в `state.particle_f`, не очищая внешние силы,
не интегрируя состояние и не обрабатывая контакты. Второй сбрасывает собственную
историю материала. Параметры и топология принадлежат компоненту.

`PhysTwinInternalForces` — одна реализация этого контракта: неизменённый
`eval_springs`, закреплённый SHA-256 исходника, и внутренние рёбра графа.
Обычные пружины Newton для этих же связей не создаются.
Другой силовой закон подключается через список `laws` без изменения viewer
или контактов. Test double с другим законом проверяет этот путь.
MPM и модели с другим состоянием/интегратором этим интерфейсом не покрываются.

Перед шагом вызывающий код очищает силы и добавляет внешние воздействия.
Solver добавляет внутренние силы и вызывает Newton. Экспоненциальный drag
применяется после ускорений и контактов с пересчётом позиции; гравитация
учитывается один раз. Входной и выходной `State` должны различаться.
CUDA Graph захватывает группу подшагов; изменяемые цели хранятся в GPU-буферах.

### Данные, эталонный режим и provenance

Данные по умолчанию находятся вне Git:
`~/PhysTwin/gaussian_output/double_stretch_sloth/.../point_cloud/iteration_10000/point_cloud.ply`
и `~/.cache/deformable-twins/phystwin-lesson/exports/double_stretch_sloth/`.
Граф проверяется по `provenance.json`; входы не перезаписываются.
Другой экспорт задаётся через `--graph`, PLY — позиционным аргументом.

```bash
# Старый эталон с управляющими точками и исходной землёй PhysTwin:
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --reference
# Статический просмотр без физики:
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --static
```

Каждый запуск Newton-примера создаёт новый каталог
`~/.cache/deformable-twins/newton-phystwin/<timestamp>/`:
`manifest.json` с версиями, хешами кода/данных, демонстрационными параметрами,
масштабом сил и преобразованием координат; для Panda — также закреплённый asset,
хеши его файлов, настройки приводов и начальная/целевая позы; `events.jsonl` для сбросов, ошибок и заданных целей суставов;
`result.json` при штатной остановке. При численной ошибке сохраняется NPZ состояния.
Исследовательские IDs равны null: запуск не является научным экспериментом.

Повторное создание экспорта в **новый** каталог (существующий не перезаписывается):

```bash
~/.venvs/phystwin-export/bin/python tutorials/export_phystwin.py \
  --source "$HOME/PhysTwin" \
  --data "$HOME/PhysTwin/data/different_types/double_stretch_sloth" \
  --optimal "$HOME/PhysTwin/experiments_optimization/double_stretch_sloth/optimal_params.pkl" \
  --checkpoint "$HOME/PhysTwin/experiments/double_stretch_sloth/train/best_199.pth" \
  --config "$HOME/PhysTwin/configs/real.yaml" \
  --output "$HOME/.cache/deformable-twins/phystwin-lesson/exports/double_stretch_sloth-v2"
```

Этот локальный экспортёр использует Python 3.11 с Torch/Open3D/PyYAML
в `~/.venvs/phystwin-export` (унаследованный Torch из `bridge_sbi`).
На другом компьютере его можно выполнять в окружении PhysTwin с этими пакетами;
в окружение viewer Torch не требуется. Загружать только доверенные pickle/checkpoint.
Экспортёр закреплён на локальную ревизию PhysTwin `2b6630528141b9cba5a7677c8b88b2129b4a8390`.
Её файл физических kernels совпадает с файлом текущего submodule.
Порядок графа восстанавливается оригинальным `_init_start`; checkpoint не содержит
хеша рёбер, поэтому совпадение с историческим графом обучения не доказано одним
лишь совпадением количества пружин.

### Проверки

```bash
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python -m unittest discover -s tests -v
~/.venvs/phystwin-newton/bin/python -m unittest discover -s tutorials -p test_phystwin_runtime.py -v
# Ручная инфраструктурная проверка полного локального графа (CUDA):
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python tests/smoke_newton_phystwin.py
# Panda: толчок, обратное воздействие на суставы, повторный сброс:
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python tests/smoke_panda_phystwin.py
# Полный автоматический цикл и выбор следующей цели:
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python tests/smoke_grasp_cycle.py
~/.venvs/phystwin-newton/bin/python -m pip check
```

Проверяются порог активности и clamp пружин, dashpot, сохранение внешних сил,
масштабирование масс/сил, drag, CPU/CUDA, замена закона, сброс и контактный
обмен импульсом в обе стороны. Полный smoke test проверяет повторный сброс,
движение игрушки от шара и изменение траектории шара из-за контактов с игрушкой.
Вариант без particle-контактов — техническая проверка обратного воздействия,
не физический baseline и не оценка качества калибровки.

Локально проверено 2026-11-03: 5 тестов нового solver и 4 теста эталонного
runtime прошли; `pip check` не обнаружил конфликтов. На RTX 5060 полный smoke
test прошёл 262 кадра, включая повторные сбросы. Его manifest и результат:
`~/.cache/deformable-twins/newton-phystwin/20261103T141841376989Z/`.
В браузере проверены перетаскивание цели, обновление Gaussian-оболочки,
пауза и сброс; скриншоты и `browser-check.json` лежат в родительском каталоге
`newton-phystwin/`. Это проверка конкретной демонстрационной сцены, не гарантия
устойчивости при произвольном воздействии.

Проверка Panda в тот же день: 7 тестов в `tests/` и 4 теста эталонного runtime
прошли, включая CPU/CUDA для контакта частицы с динамическим шарниром и
ограничение усилия PD. Полный прогон на 212 кадров с повторным сбросом сохранён в
`~/.cache/deformable-twins/newton-phystwin/20261103T144311390163Z/`.
Отключение только контактов Panda с частицами при тех же командах изменило
конечные координаты частиц максимум на 0,07755 м и один из углов руки на
0,000199 рад. Это техническая проверка двустороннего контакта, не качество
физической калибровки. В браузере проверены отображение руки и гауссиан,
кнопка толчка, пауза и сброс; снимки в `newton-phystwin/panda-browser/`.
В этой первоначальной версии расчёт был медленнее реального времени;
результаты ускоренной версии приведены ниже. Надёжность полного захвата не подтверждена.

## Ускоренная симуляция Panda

По умолчанию viewer использует `MultiRate`: частицы и исходный kernel пружин
PhysTwin работают с прежним шагом 50 мкс. Контактные силы пересчитываются на
каждом таком шаге. Геометрия контактов и динамика руки обновляются каждые
20 подшагов (1 мс), матрица масс — раз в восемь шагов руки; последний неполный
интервал кадра обрабатывается отдельно. Обратная сумма контактных импульсов
передаётся руке через среднюю силу. Положение/нормаль контактной поверхности
внутри этого интервала фиксированы: это численное приближение, не идентичная
траектория исходного solver. Первое обновление матрицы масс выполняется на каждом
кадре, включая кадр после сброса. Переполнение буфера контактов останавливает расчёт.

```bash
# Ускоренный вариант (по умолчанию):
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --viewer gl
# Исходный расчёт всех компонентов каждые 50 мкс для сравнения:
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --viewer gl --accurate
```

IK также выполняется на GPU Newton (аналитический якобиан, 48 итераций LM),
чтобы подготовка новой цели не останавливала окно на несколько секунд.
Деформация Gaussian-оболочки перенесена на GPU, включая локальный SVD и
взвешенное преобразование центров. Проверка против прежней CPU-реализации
проходит с абсолютным допуском 2 мкм. Граф пружин, массы, контактные коэффициенты
и число физических частиц не сокращены. Режим coupling и его интервалы записаны
в manifest. Ускорение не является валидацией поведения произвольных сцен.

```bash
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python tests/compare_coupling.py
PYTHONPATH=src ~/.venvs/phystwin-newton/bin/python tests/smoke_grasp_cycle.py --fast
```

2026-11-03: 12 unit tests прошли, включая CPU/CUDA обмен импульсом при разных
частотах и GPU-привязку гауссиан. Полный цикл проверен в оконном прогоне ниже.
На 60 одинаковых командах медиана времени физики снизилась с 601,8 до 23,3 мс;
RMS-разница координат частиц — 0,018 мм, максимум — 0,70 мм.
Артефакты сравнения в кэше: `newton-phystwin/20261103T151731181403Z/`
и `newton-phystwin/20261103T151809041000Z/`. Это ограниченная техническая проверка,
а не доказательство эквивалентности на всех взаимодействиях.

В окончательном OpenGL-прогоне на RTX 5060 (400 кадров, 1280×900, 40 000
отображаемых центров гауссиан) средняя частота — **32,36 FPS**, медиана кадра —
**30,53 мс**, 95-й перцентиль — **31,96 мс**. Замер включает подготовку целей,
полный первый захват, подъём, сброс и начало второго. Симуляция продвинулась на
13,34 с за примерно 12,36 с работы цикла. Артефакты с manifest, событиями,
`result.json` и снимком: `newton-phystwin/20261103T151637849600Z/`.
Это измерение конкретной сцены/оборудования, не гарантия минимального FPS на всех кадрах.

## Отдельное окно Newton OpenGL

```bash
~/.venvs/phystwin-newton/bin/python -m pip install -r environments/requirements-newton-gl.txt
~/.venvs/phystwin-newton/bin/python tutorials/view_phystwin_gaussians.py --viewer gl --seed 0
```

Тот же автоматический цикл Panda открывается в `Newton ViewerGL`. Штатные
Pause / Single step / Reset управляют симуляцией; закрытие окна завершает процесс.
В панели показана текущая фаза. Красная сфера отмечает случайную цель захвата.
Игрушка отображается цветными сферами в центрах гауссиан с прежней деформационной
привязкой: это не прозрачный Gaussian splatting браузерного viewer.
Переключатель частиц в Rendering показывает дополнительно физические частицы.

Режим GL пока предназначен для автоматического Panda-демо; `--sphere`,
`--reference` и `--static` работают в браузере (`--viewer viser`, по умолчанию).
Смена viewer сама по себе не меняет физику; по умолчанию Panda использует ускоренную схему ниже. Нужен доступ к локальному дисплею
и OpenGL. Для ограниченной проверки `--viewer gl --frames 3` закрывает окно после
трёх кадров и сохраняет `gl-frame.png` рядом с manifest; `--headless --frames 3`
предназначен для проверки без видимого окна.

Проверено 2026-11-03 на хосте: оконный и headless GL-запуски по три кадра
завершились без ошибки; скриншот оконного режима с панелью управления просмотрен.
Артефакты: `newton-phystwin/20261103T150139173112Z/` (окно) и
`newton-phystwin/20261103T150201904155Z/` (headless) внутри каталога кэша выше.
Установка дополнительных GL-пакетов прошла `pip check`.

## Распределение обученных параметров пружин

```bash
~/.venvs/phystwin-newton/bin/python tutorials/analyze_phystwin_springs.py
```

Скрипт читает проверенный экспорт графа и создаёт новый каталог
`~/.cache/deformable-twins/spring-analysis/<timestamp>/` с PNG/PDF,
таблицей узлов `nodes.csv` и `manifest.json` (входные хеши, методика и числа).
Параметр Y показан после порога активности и clamp; отдельно учитываются
внутренние связи и связи управляющих точек. Начальное Y₀ взято из конфигурации
первого этапа оптимизации. Узлы окрашены по среднему Y соседних внутренних связей,
чтобы количество соседей само по себе не делало узел «жёстким».

Это описательный анализ одного checkpoint. Он не измеряет причинную важность
узлов: для этого нужны отдельный протокол вмешательств в параметры и оценка
на отложенных взаимодействиях. Y здесь — масштаб осевой силы; эквивалентная
продольная жёсткость пружины равна Y/L₀, а не самому Y.

### Проверка автоматического захвата

2026-11-03: 10 автоматических тестов прошли. Полный smoke test цикла с seed 0
прошёл 364 кадра, включая автоматический сброс и выбор следующей точки.
Артефакты: `~/.cache/deformable-twins/newton-phystwin/20261103T145243607890Z/`.
В первой попытке максимальный подъём частицы составил 0,10735 м, средний —
0,01005 м; 14,34% частиц поднялись более чем на 2 см. Это частичный подъём,
а не подтверждение отрыва всей игрушки от стола. Для следующей случайной точки
IK не нашёл допустимую последовательность; попытка записана и сброшена.
Успешный полный захват произвольной точки не гарантируется.

Подъём на 30 см проверен 2026-11-03: 3 теста последовательности прошли,
полный ускоренный прогон завершил 378 кадров, сброс и выбор второй цели.
В первой попытке удержать игрушку до конца подъёма не удалось: ни одна частица
не осталась выше исходного положения более чем на 2 см. Это не успешный захват.
