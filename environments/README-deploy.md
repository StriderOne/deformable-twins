# Развертывание через SSH

Целевая машина: `roman@192.168.3.192`, репозиторий
`/home/roman/deformable-twins`. Контейнер `train` содержит существующую
VI/MAP-калибровку PhysTwin; `viewer` — Newton, Viser и OpenGL.
Это подготовка инфраструктуры для [[Q-0001]], без запуска научного эксперимента
или обучения политики. Параметры, сплиты и критерии экспериментов остаются
в `experiments/` и `registry/`.

## Требования и сборка

Linux x86_64, NVIDIA GPU, Docker с NVIDIA Container Toolkit, Git, Python 3,
`curl`; для окна — X11/XWayland и `xauth`. Проверенная конфигурация сервера:
RTX 5080 16 GiB, драйвер 595.99.02, Docker 28.5.2.
Хостовые Python/Conda не используются для обучения. Данные и кэши находятся
вне Git; оставьте место для архивов, распакованных данных и Docker-слоёв.

Выполнять на сервере из корня репозитория:

```bash
ssh roman@192.168.3.192
cd ~/deformable-twins
git submodule update --init third_party/PhysTwin
docker build -f environments/Dockerfile.train -t deformable-twins/train:cu128-v1 environments
docker build -f environments/Dockerfile.viewer -t deformable-twins/viewer:cu128-v1 environments
```

Базовые образы закреплены по digest; версии прямых Python-зависимостей
закреплены в Dockerfile и существующих requirements. Транзитивные пакеты
не имеют общего lock-файла; для конкретной сборки сохраняйте `pip freeze`
и image ID. Существующий `environments/Dockerfile` с Isaac Lab и
`Dockerfile.isaac-sim` остаются отдельными окружениями. На целевом сервере
загрузка закреплённого Isaac Lab из `nvcr.io` вернула HTTP 403;
новые контейнеры используют Docker Hub и не требуют NGC.
Полный RTX-renderer Isaac Sim в эти два образа не входит.

## Данные PhysTwin

```bash
python3 tools/download_phystwin.py
```

Источник — [официальный PhysTwin на Hugging Face](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/tree/05665bcd0edad8f069c7def9cbb58d2eb75d68d0).
Версия, размеры архивов, лицензия и ожидаемые SHA-256 записаны в
[`phystwin-release-v1.json`](../datasets/phystwin-release-v1.json).
По умолчанию скачиваются `data`, `experiments_optimization`, `experiments`
и `gaussian_output`: наблюдения с готовой обработкой, опубликованные результаты
калибровки и Gaussian assets. Это около 10.2 GB сжатых данных.
Архивы с результатами являются опубликованными артефактами авторов,
а не результатом нашего обучения. Дополнительные сцены необязательны:

```bash
python3 tools/download_phystwin.py --archives additional_data.zip
```

По умолчанию корень — `~/PhysTwin`, можно задать `--root /path/to/PhysTwin`.
Архивы и записи происхождения остаются в `~/PhysTwin/.downloads/`.
После обрыва повторите команду: `.part` докачивается, SHA-256 и ZIP CRC
проверяются перед публикацией каталога. Существующие каталоги без записи
загрузки не перезаписываются. Повторный запуск пропускает завершённые архивы;
проверку изменений уже распакованных файлов выполняют манифесты входных данных.

`container.sh` монтирует данные только для чтения в `/home/homa/PhysTwin`,
с `HOME=/home/homa` внутри контейнера. Это сохраняет абсолютные пути
существующих манифестов `datasets/sloth-*.json` без изменения исходных записей.
На хосте данные остаются в `/home/roman/PhysTwin`.
При другом каталоге задайте `PHYSTWIN_ROOT=/path/to/PhysTwin`.
Код монтируется в `/workspace`; новые run manifests пишутся в хостовый `runs/`,
артефакты — в `~/.local/share/deformable-twins`, кэши — в
`~/.cache/deformable-twins/container-{train,viewer}`. UID/GID совпадают с хостом.

## Проверка окружения

```bash
environments/container.sh train python -m pip check
environments/container.sh train python -c 'import torch, warp as wp; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu()); wp.init(); assert wp.get_device("cuda:0").is_cuda'
environments/container.sh train python -m unittest discover -s tests -p test_vi_posterior.py -v
environments/container.sh viewer python -m pip check
```

Интерактивная оболочка: `environments/container.sh train`.
Эти образы не устанавливают upstream preprocessing stack
TRELLIS/Grounded-SAM и CUDA-расширения оригинального renderer PhysTwin:
наша калибровка использует уже обработанные наблюдения и AST-адаптер.

## Визуализация в браузере через SSH

На сервере, статический просмотр опубликованного Gaussian asset без физики:

```bash
environments/container.sh viewer
```

На клиенте откройте SSH-туннель:

```bash
ssh -N -L 8080:127.0.0.1:8080 roman@192.168.3.192
```

Откройте `http://localhost:8080`. Docker публикует порт только на loopback
сервера; Viser внутри контейнера слушает `0.0.0.0`. Для этого варианта не нужны `DISPLAY` и открытые внешние порты.
Остановка — Ctrl+C в терминале viewer. Viser передаёт Gaussian-сцену в браузер;
качество и скорость зависят также от GPU клиента.

Для динамической демонстрации сначала создайте производный граф в **новом**
каталоге, сохранив provenance. Это экспорт без запуска симуляции:

```bash
environments/container.sh train python tutorials/export_phystwin.py \
  --source /workspace/third_party/PhysTwin \
  --data /home/homa/PhysTwin/data/different_types/double_stretch_sloth \
  --optimal /home/homa/PhysTwin/experiments_optimization/double_stretch_sloth/optimal_params.pkl \
  --checkpoint /home/homa/PhysTwin/experiments/double_stretch_sloth/train/best_199.pth \
  --config /workspace/third_party/PhysTwin/configs/real.yaml \
  --output /home/homa/.local/share/deformable-twins/exports/double_stretch_sloth
environments/container.sh viewer python tutorials/view_phystwin_gaussians.py \
  --host 0.0.0.0 --graph /home/homa/.local/share/deformable-twins/exports/double_stretch_sloth
```

Граф поддерживает существующую демонстрацию Panda/Newton; это симуляция,
не подключение к физическому роботу и не оценка sim-to-real.

## Окно OpenGL из контейнера

В графическом терминале сервера с корректными `DISPLAY` и `XAUTHORITY`:

```bash
environments/container.sh viewer --x11 glxinfo -B
environments/container.sh viewer --x11 python tutorials/view_phystwin_gaussians.py \
  --viewer gl --graph /home/homa/.local/share/deformable-twins/exports/double_stretch_sloth
```

Прокидываются X11 socket, временный cookie и возможности NVIDIA
`compute,utility,graphics,display`, как описано в
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
`xhost +` не используется. Cookie удаляется при завершении.

Оболочка поддерживает также `ssh -Y` и `DISPLAY=localhost:…` через host network,
но OpenGL/CUDA interoperability через удалённый GLX зависит от X-сервера
клиента; для удалённого просмотра используйте браузерный вариант выше.

## Перед обучением

Исторические планы могут ссылаться на отсутствующие checkpoints и артефакты
прежних запусков; загрузка релиза не восстанавливает их. Выберите согласованный
план и новый run ID. Перед каждым научным запуском разрешите все
`Human input required` по `AGENTS.md`, затем проверьте рабочее дерево:

```bash
git status --porcelain=v1 --untracked-files=all --ignore-submodules=none
git rev-parse HEAD
```

Продолжать можно только при пустом выводе первой команды. Для существующего плана E-0001
точка входа — `environments/container.sh train python -m lab.calibration.phystwin_vi`;
параметры смотрите через `--help`. Подготовка сервера не запускает этот план.
Наличие контейнеров и датасета не означает, что обучение политики или
физическая оценка выполнены.


## Проверено на целевой машине 2026-11-19

- Оба Docker-образа собраны; `pip check` проходит в обоих.
- PyTorch 2.9.1+cu128 и Warp 1.17.0 видят RTX 5080; проверены CUDA allocation
  и вычисление Torch, без научного прогона.
- 4 теста загрузчика и 8 тестов `test_vi_posterior.py` проходят.
- Четыре стандартных архива скачаны, SHA-256 и ZIP CRC проверены.
  Все зарегистрированные входы `sloth-v1.json` и `sloth-double-lift-v1.json`
  совпадают по SHA-256; `read_plan` для E-0001 проходит.
- `glxinfo -B` через `container.sh viewer --x11` подтверждает direct rendering,
  OpenGL 4.6 и NVIDIA RTX 5080 в XWayland-сессии сервера.
- Статический Viser загрузил 233293 Gaussian splats; HTTP 200 проверен
  на сервере и с клиента через SSH-туннель. Интерактивное управление камерой
  в браузере вручную не проверялось. Проверочный viewer остановлен.
- Производный граф экспортирован в
  `~/.local/share/deformable-twins/exports/double_stretch_sloth`:
  6517 частиц, 53166 пружин, 192 кадра. Симуляция и обучение не запускались.

Логи, image IDs, source hashes и `pip freeze` конкретных сборок:
`~/.cache/deformable-twins/deployment/`. Машинная сводка — `validation.json`.
