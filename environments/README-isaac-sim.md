# PhysTwin + Panda в отдельном Isaac Sim RTX контейнере

Это дополнительный renderer существующей Newton-сцены: робот и стол — USD-меши,
игрушка — native `ParticleField3DGaussianSplat` из расширения Kit. Силы PhysTwin, контакты
и цикл случайного захвата остаются в Newton. USD не содержит второй физической
сцены. Установки Isaac, Conda и venv на хосте не изменяются.

## Запуск окна на хосте

Из корня репозитория:

```bash
./environments/run-isaac-sim.sh
```

Скрипт собирает отдельный образ из `Dockerfile.isaac-sim`, передаёт GPU,
`DISPLAY`, X11-сокет и временный cookie с правами 600. Cookie удаляется при
завершении, контейнер запускается с `--rm`, UID/GID 1000:1000. Не используются
`xhost +`, privileged, Docker socket или host network. Исходники и входные
данные монтируются read-only. Команда принимает лицензию Isaac Sim через
`ACCEPT_EULA=Y`; анонимная телеметрия и privacy/usage отключены настройками Kit.
Используется `NVIDIA_DRIVER_CAPABILITIES=all`: на этой машине ограниченный
набор capabilities не передавал NGX, необходимую для DLSS/денойзинга RTX.

Нужны Docker с NVIDIA Container Toolkit, X11 и `xauth`. Данные по умолчанию:
локальная Gaussian PLY `~/PhysTwin/gaussian_output/double_stretch_sloth/...`
и экспорт `~/.cache/deformable-twins/phystwin-lesson/exports/double_stretch_sloth`.
Можно указать другие пути:

```bash
PHYSTWIN_PLY=/absolute/point_cloud.ply PHYSTWIN_EXPORT=/absolute/export \
  ./environments/run-isaac-sim.sh
```

Окно 1280×720, одна сцена, не более 40 000 гауссиан, SH степени 0, без Frame
Generation. Как в существующем Viser-примере, поддерживается изотропная оболочка.
Число физических частиц/пружин и шаг 50 мкс не меняются. Кнопки Pause / Resume
и Reset находятся в панели PhysTwin. Закрытие окна завершает процесс.

```bash
# Ограниченная проверка окна, screenshot и result.json:
./environments/run-isaac-sim.sh --frames 100
# Остановка из другого терминала:
docker stop phystwin-isaac-rtx
```

Кэши и логи: `~/.cache/deformable-twins/isaac-sim/`;
manifest, события и screenshot: `cache/deformable-twins/newton-phystwin/<timestamp>/`
внутри этого каталога. Для первой сборки скрипт требует не менее 60 GiB свободного
места. Не запускайте `docker pull` и `docker build` параллельно: это может
удвоить временное место при распаковке. Перед запуском и во время работы
проверяется резерв 10 GiB; при его нарушении расчёт завершается с ошибкой.
Чужие данные автоматически не удаляются.

## RTX 5060 и проверка

Целевая машина: RTX 5060, 8 GiB VRAM, драйвер 580.159.04. Драйвер новее
проверенного для Isaac Sim 6.0 Linux 580.95.05, но память ниже официального
минимума 16 GiB. Ограниченная сцена требует фактической проверки; 30 FPS
и пригодность для больших сцен не гарантируются.

Образ 6.0.0 закреплён по digest linux/amd64. Newton 1.6.1 и Warp 1.17.0 закреплены в Dockerfile; встроенные пути
Kit перенаправлены на эти же пакеты, чтобы избежать смешивания версий.
Встроенные NumPy 2.3.1 и SciPy 1.17.0 сохранены: замена SciPy ломала
дедуплицированные ссылки на библиотеки NumPy. Другие сценарии Isaac Sim
в этом специализированном образе не проверялись. При изменении Dockerfile пересоберите образ явно:

```bash
docker build -t deformable-twins/isaac-sim:6.0.0 - < environments/Dockerfile.isaac-sim
```

Источники: [контейнер и GUI](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html),
[требования](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html),
[USD Gaussian fields и RTX](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html).

Кэш UJITSO необходим для native Gaussian-геометрии. Для меняющихся каждый кадр
гауссиан его бюджет ограничен 512 MB, размер файлов — 64 MB, очистка — раз в
секунду. Это параметры кэша, не изменение физики или числа гауссиан. Кэши
драйвера и Warp сохраняются. Первый запуск может несколько минут компилировать
шейдеры; последующие используют кэш. Настройки записаны в manifest.

Пауза и сброс выполняются кнопками панели **PhysTwin — Newton physics**.
Стандартная кнопка Play на timeline Isaac не управляет внешней физикой Newton.
Запуск без окна поддерживается через `--headless --frames 100` (оконный режим
используется по умолчанию).

## Фактическая проверка 2026-11-03

На RTX 5060 8 GiB окно из контейнера открывается через X11. Проверены совместное
отображение мешей Panda и Gaussian splats, деформация, первый цикл подъёма,
автоматический сброс и выбор следующей цели. Прогон **500 кадров**, код выхода 0;
12 тестов физики внутри контейнера прошли. Синтаксис Bash и Python проверен.

При 1280×720 и 40 000 гауссиан: средняя частота **16,90 FPS**, медиана кадра
57,92 мс, p95 60,59 мс. Это измерение этой сцены; 30 FPS не достигнуты.
Первый подъём не удержал игрушку — успешный захват не заявляется.
Кнопки Pause / Reset в интерфейсе вручную не проверялись.

Manifest, screenshot, события, результаты и логи:
`~/.cache/deformable-twins/isaac-sim/cache/deformable-twins/newton-phystwin/20261103T164919128686Z/`.
В manifest сохранены хеши исходников/PLY/графа, настройки физики и renderer,
сборка Isaac Sim, версии USD/SciPy и ID образа контейнера.

## Режим качества

Isaac теперь по умолчанию использует все гауссианы PLY без отсечения по
прозрачности (`--max-gaussians 0`) и DLAA при исходном разрешении 1280×720.
Для текущей игрушки это 233 293 гауссианы. Viser и GL сохраняют прежний
лимит 40 000. Цвет пока SH0; полные направленные SH не перенесены.

```bash
./environments/run-isaac-sim.sh
# Меньше гауссиан, если важнее скорость (DLAA сохраняется):
./environments/run-isaac-sim.sh --max-gaussians 40000
```

Настройка DLAA: [NVIDIA RTX renderer](https://docs-prod.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html).

Проверка полного варианта с DLAA: 100 кадров, выход 0, 9.43 FPS;
пиковая общая память GPU 3264 MiB (включая другие приложения).
Количественное улучшение качества не измерялось.
Артефакты: `/home/homa/.cache/deformable-twins/isaac-sim/cache/deformable-twins/newton-phystwin/20261103T170425045878Z`.
