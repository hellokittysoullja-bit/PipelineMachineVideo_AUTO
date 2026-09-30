# Источники фикстур golden media-selection теста

Все 4 исходных фото — реальные фотографии с Wikimedia Commons, лицензии
разрешают переиспользование при указании авторства (нет CC0/PD — авторство
обязательно). Уменьшены до max 768px по длинной стороне и пережаты (JPEG
quality ~82) для размера в репозитории — содержание кадра не изменено.

## sword.jpg
- Оригинал: "Albion Cluny Medieval Sword 13"
- Автор: Søren Niedziella (Дания)
- Лицензия: CC BY 2.0
- Источник: https://commons.wikimedia.org/wiki/File:Albion_Cluny_Medieval_Sword_13_(6092968660).jpg

## stainedglass.jpg
- Оригинал: "St Vitus Cathedral - Stained glass (retouch)"
- Автор: Oryg. Pudelek, Mody. Albertus teolog
- Лицензия: CC BY-SA 4.0
- Источник: https://commons.wikimedia.org/wiki/File:St_Vitus_Cathedral_-_Stained_glass_(retouch).jpg

## pizza.jpg
- Оригинал: "Vegetarian Pizza"
- Автор: Petar Milošević
- Лицензия: CC BY-SA 4.0
- Источник: https://commons.wikimedia.org/wiki/File:Vegetarian_Pizza.jpg

## meeting.jpg
- Оригинал: "Meeting British business representatives"
- Автор: Foreign and Commonwealth Office (Great Britain)
- Лицензия: OGL v1.0 (UK Open Government Licence)
- Источник: https://commons.wikimedia.org/wiki/File:Meeting_British_business_representatives_(5430620349).jpg

## sword_near_dup.jpg / sword_degraded.jpg
Производные от sword.jpg (см. выше), сгенерированы локально в этом
репозитории (лёгкий кроп 2% с краёв — имитация того же фото у другого
стокового источника; и Gaussian blur — имитация плохого/размытого
кандидата) — не самостоятельные внешние источники, лицензия та же, что
у sword.jpg (CC BY 2.0, Søren Niedziella).

## katana.jpg
- Источник: Pexels, https://www.pexels.com/photo/woman-in-floral-kimono-holding-katana-in-hand-7778825/
- Автор: cottonbro studio
- Лицензия: Pexels License (свободное использование, изменение, без
  обязательной атрибуции — https://www.pexels.com/license/)
- Уменьшено до 768px/JPEG q82, как остальные фикстуры выше.

## euro_sword_2.jpg
- Источник: Pexels, https://www.pexels.com/photo/intricately-designed-historical-sword-close-up-31350049/
- Автор: Blackcurrant Great
- Лицензия: Pexels License (см. katana.jpg выше)
- Уменьшено до 768px/JPEG q82, как остальные фикстуры выше.

katana.jpg/euro_sword_2.jpg — фикстуры для калибровки VISUAL_DOMAIN_GUARDS
(scripts/pipeline_smart.py: guard "east_asian_sword" — см. коммит и
tests/test_media_selection_golden.py) — та же пара классов
(восточноазиатский vs европейский меч), что реально спутал Pexels-подбор
на живом эпизоде 01_ves-mecha.

Использование: только как тестовые фикстуры для CLIP-based media-selection
теста (tests/test_media_selection_golden.py), не для производства видео.

## sword_snow.jpg
- Источник: Pexels, скачано пайплайном (stock_fetch/pexels_photo) по
  тематическому запросу про меч — конкретный ID фото кэш не сохраняет,
  поэтому прямая ссылка недоступна; Pexels License распространяется на
  весь контент Pexels (свободное использование и изменение, атрибуция не
  обязательна — https://www.pexels.com/license/).
- Уменьшено до 768px/JPEG q82, как остальные фикстуры выше.

Фикстура для двух регрессий на РЕАЛЬНОМ кадре (не синтетике):
classify_domain() должен давать "snow" (tests/test_look_reference.py) и
detect_face_anchor() не должен ложно находить лицо на этом кадре
(tests/test_parse.py). Раньше оба теста брали это фото glob-ом из
videos/*/temp_smart/pexels_cache/ — то есть из изменчивого кэша рабочей
копии, где имя файла не идентифицирует фотографию (см. комментарии в
самих тестах).
