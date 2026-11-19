---
id: phystwin-sloth-additional
type: dataset
status: available
---

# Дополнительные взаимодействия с ленивцем

[[Q-0001]] · [[H-0001]] · [[E-0006]]

## Observation
Из официального PhysTwin data.zip скачаны double_lift_sloth, single_lift_sloth
и single_push_sloth. Основные данные находятся в
`/home/homa/PhysTwin/data/different_types/<case>/`; RGB-видео — `color/{0,1,2}.mp4`.
Также извлечены соответствующие авторские derived-наборы different_types_human_mask
и gaussian_data. Существующие данные double_stretch_sloth не изменялись.

[Manifest](../../datasets/sloth-additional-v1.json) содержит pinned revision источника,
SHA256 всех извлечённых файлов, ZIP CRC, metadata, исходные splits и проверку видео.
Все выбранные ZIP members прошли CRC; количество кадров всех видео совпало с metadata.
Полный архив не скачивался: использованы HTTP ranges. Его опубликованный SHA256
записан как reference, а не как локально проверенный checksum полного архива.

## Ограничения
Эти последовательности не использовались в выполненных VI runs.
Перенос параметров между записями требует проверки соответствия геометрии и пружин:
совпадение объекта само по себе не означает совпадения spring IDs.
Сессии съёмки, полная provenance и условия лицензирования независимо не проверены.
