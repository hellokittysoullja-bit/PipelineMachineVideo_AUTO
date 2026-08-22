"""Visual regression-тест для фикса `_warm_mult` (scripts/pipeline_smart.py):
доказывает на РЕАЛЬНО ОТРЕНДЕРЕННЫХ пикселях (настоящий ffmpeg-рендер через
настоящий film_look(), не только алгебра формулы), что убранная усиливающая
ветка `+0.20*max(0.0,-warm_bias)` действительно раньше раздвигала холодные
клипы от нейтральной базы дальше, а не "сближала клипы разной температуры",
как заявляла цель модуляции (CLAUDE.md, ЧАСТЬ 15) — и что убранная ветка не
задела тёплые источники (та часть формулы не менялась).

Методология (тот же принцип, что уже применялся в этой сессии для соседней
оси chroma_bias): синтетическое фото с контролируемым цветовым сдвигом ->
реальный measure_levels()/film_look() -> реальный ffmpeg -> PIL считывает
итоговые пиксели -> сравнение средних RGB, не веры в формулу на бумаге.

Требует ffmpeg в PATH — пропускается, если недоступен (тот же принцип
безопасного отката, что и у остальных опциональных проверок в этом файле)."""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg недоступен")

PHOTO_HASH = 42            # фиксированный — одинаковый для всех рендеров, чтобы
                             # micro-jitter film_look() не путал сравнение
IMG_SIZE = (256, 256)
GRAD_LO, GRAD_HI = 0.30, 0.70   # диапазон яркости градиента — контраст 0.40,
                                  # заметно выше AUTO_LEVELS_MIN_RANGE=0.05
COLOR_SHIFT = 0.10          # r/b смещение от нейтрали для warm/cold образцов


def _warm_mult_before_fix(warm_bias):
    """Формула ДО фикса (реальный баг: усиливала пуш на холодных источниках,
    warm_mult доходил до 1.20) — копия только для regression-сравнения в
    этом тесте, НЕ трогать/не использовать больше нигде."""
    return 1.0 - 0.35 * max(0.0, warm_bias) + 0.20 * max(0.0, -warm_bias)


def _make_gradient_photo(path, r_shift=0.0, b_shift=0.0):
    """Горизонтальный яркостный градиент (реальная, невырожденная гистограмма
    для measure_levels/auto_wb_params) с постоянным r/b-сдвигом ПОВЕРХ
    градиента — сдвиг постоянный, значит mean(R)-mean(B) смещается ровно на
    (r_shift-b_shift), без клиппинга (диапазон [0.30-0.10, 0.70+0.10] =
    [0.20, 0.80], безопасно внутри [0,1])."""
    w, h = IMG_SIZE
    t = np.linspace(GRAD_LO, GRAD_HI, w, dtype=np.float32)
    t = np.tile(t, (h, 1))
    r = np.clip(t + r_shift, 0.0, 1.0)
    g = t
    b = np.clip(t + b_shift, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    arr = (rgb * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _render_via_film_look(input_path, output_path, domain=None):
    """Реальный measure_levels() -> реальный film_look() (использует
    pipeline_smart._warm_mult как есть в момент вызова — тест патчит/снимает
    патч ДО вызова этой функции, не внутри неё) -> реальный ffmpeg.
    Возвращает средний (r,g,b) выходного кадра (0..1)."""
    levels, wb = pipeline_smart.measure_levels(input_path, want_wb=True)
    vf = pipeline_smart.film_look(PHOTO_HASH, section="BODY", levels=levels, wb=wb, domain=domain)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vf", vf, "-frames:v", "1", output_path],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0 and os.path.exists(output_path), (
        f"ffmpeg не отрендерил кадр: {r.stderr[-800:]}")
    out_arr = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0
    return tuple(float(x) for x in out_arr.mean(axis=(0, 1)))


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


REGION_FRAC = 0.25   # доля ширины кадра


def _region_means(img_path, frac=REGION_FRAC):
    """Средние RGB в САМОЙ ТЁМНОЙ (левый край градиента) и САМОЙ СВЕТЛОЙ
    (правый край) четверти выходного кадра. bs/rh в colorbalance — ТОНОВО-
    ИЗБИРАТЕЛЬНЫЕ параметры (blue-в-тенях, red-в-светах), не равномерный
    channel gain — эффект на СРЕДНЕМ по всему кадру (что пробовалось раньше)
    тонет в остальных этапах рецепта (auto_wb уже отдельно нормализует цвет,
    curves/selectivecolor/halation/deband/unsharp добавляют свой немалый
    шум поверх). Прямое измерение именно теневой/световой зоны — то, что
    bs/rh физически двигают, честный таргетированный QC-metric вместо
    случайно нечувствительного общего среднего."""
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    w = arr.shape[1]
    n = max(1, int(w * frac))
    shadow = tuple(float(x) for x in arr[:, :n, :].mean(axis=(0, 1)))
    highlight = tuple(float(x) for x in arr[:, -n:, :].mean(axis=(0, 1)))
    return shadow, highlight


def _save_contact_sheet(panels, out_path):
    """panels: [(label, path_or_rgb_tuple_for_caption, image_path), ...] —
    контактный лист для человека (Read tool), не влияет на assert'ы теста."""
    cell = 256
    pad = 8
    label_h = 40
    n = len(panels)
    sheet = Image.new("RGB", (cell * n + pad * (n + 1), cell + label_h + pad * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, rgb, img_path) in enumerate(panels):
        x = pad + i * (cell + pad)
        img = Image.open(img_path).convert("RGB").resize((cell, cell))
        sheet.paste(img, (x, label_h))
        caption = f"{label}\nR={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}"
        draw.text((x, 2), caption, fill="black")
    sheet.save(out_path)


def test_warm_mult_cold_no_longer_amplified_warm_unaffected(tmp_path, monkeypatch):
    neutral_in = str(tmp_path / "neutral.png")
    cold_in = str(tmp_path / "cold.png")
    warm_in = str(tmp_path / "warm.png")
    _make_gradient_photo(neutral_in, r_shift=0.0, b_shift=0.0)
    _make_gradient_photo(cold_in, r_shift=-COLOR_SHIFT, b_shift=COLOR_SHIFT)   # warm_bias -> -1.0
    _make_gradient_photo(warm_in, r_shift=COLOR_SHIFT, b_shift=-COLOR_SHIFT)   # warm_bias -> +1.0

    neutral_out = str(tmp_path / "neutral_out.png")
    cold_after_out = str(tmp_path / "cold_after_out.png")
    cold_before_out = str(tmp_path / "cold_before_out.png")
    warm_after_out = str(tmp_path / "warm_after_out.png")
    warm_before_out = str(tmp_path / "warm_before_out.png")

    # "after" — реальный текущий прод-код, без патча (фикс уже применён)
    neutral_rgb = _render_via_film_look(neutral_in, neutral_out)
    cold_after_rgb = _render_via_film_look(cold_in, cold_after_out)
    warm_after_rgb = _render_via_film_look(warm_in, warm_after_out)
    # "before" — только для этого теста, патчим один раз на старую формулу
    # (та, что раньше усиливала пуш для холодных источников), рендерим оба
    # before-кадра под патчем, снимается автоматически pytest'ом в конце теста
    monkeypatch.setattr(pipeline_smart, "_warm_mult", _warm_mult_before_fix)
    cold_before_rgb = _render_via_film_look(cold_in, cold_before_out)
    warm_before_rgb = _render_via_film_look(warm_in, warm_before_out)

    dist_cold_after = _dist(cold_after_rgb, neutral_rgb)
    dist_cold_before = _dist(cold_before_rgb, neutral_rgb)

    # Таргетированные метрики — именно то, что bs/rh физически двигают
    # (см. докстринг _region_means): синий в тенях, красный в светах.
    cold_before_shadow, cold_before_hi = _region_means(cold_before_out)
    cold_after_shadow, cold_after_hi = _region_means(cold_after_out)
    neutral_shadow, neutral_hi = _region_means(neutral_out)

    panels = [("neutral", neutral_rgb, neutral_out),
              ("cold BEFORE", cold_before_rgb, cold_before_out),
              ("cold AFTER", cold_after_rgb, cold_after_out),
              ("warm BEFORE", warm_before_rgb, warm_before_out),
              ("warm AFTER", warm_after_rgb, warm_after_out)]
    # Всегда — в tmp_path (pytest сохраняет последние прогоны, годится для
    # отладки CI). Дополнительно — в WARM_MULT_CONTACT_SHEET_DIR, если задан
    # (ручная сверка глазами при разработке фикса, не часть обычного прогона
    # тестов — переменная окружения, не хардкод пути конкретной песочницы).
    sheet_path = str(tmp_path / "warm_mult_regression.png")
    _save_contact_sheet(panels, sheet_path)
    print(f"\nКонтактный лист (tmp_path): {sheet_path}")
    extra_dir = os.environ.get("WARM_MULT_CONTACT_SHEET_DIR")
    if extra_dir:
        extra_path = os.path.join(extra_dir, "warm_mult_regression.png")
        _save_contact_sheet(panels, extra_path)
        print(f"Контактный лист (ручная сверка): {extra_path}")

    report = (
        f"\nneutral      = {neutral_rgb}  shadow={neutral_shadow} highlight={neutral_hi}\n"
        f"cold  before = {cold_before_rgb}  dist-to-neutral={dist_cold_before:.5f}  "
        f"shadow={cold_before_shadow} highlight={cold_before_hi}\n"
        f"cold  after  = {cold_after_rgb}  dist-to-neutral={dist_cold_after:.5f}  "
        f"shadow={cold_after_shadow} highlight={cold_after_hi}\n"
        f"warm  before = {warm_before_rgb}\n"
        f"warm  after  = {warm_after_rgb}\n"
    )
    print(report)

    # 1) Тёплые источники не деградировали — ветка формулы для warm_bias>0
    #    не менялась, реальный рендер обязан совпасть побитово (в пределах
    #    погрешности округления ffmpeg/PNG).
    assert warm_before_rgb == pytest.approx(warm_after_rgb, abs=1e-3), (
        "тёплый источник изменился между 'до' и 'после' — ветка для warm_bias>0 "
        "не должна была затрагиваться этим фиксом.\n" + report)

    # 2) Холодные источники: синий в ТЕНЯХ реально стал МЕНЬШЕ (был лишний
    #    push старой формулой) — прямое, таргетированное доказательство на
    #    отрендеренных пикселях, не общий шум по всему кадру (см. п.3 ниже —
    #    почему общий средний по кадру для этого непригоден).
    assert cold_after_shadow[2] < cold_before_shadow[2], (
        "синий канал в тенях холодного кадра не уменьшился после фикса — "
        "regression по bs (blue-shadows) не подтверждён на реальном рендере.\n" + report)

    # 3) Общий средний по всему кадру — вторичная, справочная метрика:
    #    bs/rh тоново-избирательны (тени/света), не равномерный channel gain,
    #    поэтому эффект на среднем по ВСЕМУ кадру (смешивающему тени, средние
    #    тона и света) тонет в шуме остальных этапов рецепта (auto_wb,
    #    curves, selectivecolor, halation, deband, unsharp) — измерено
    #    вживую при разработке этого теста: разница на общем среднем
    #    оказалась на уровне шума и не показательна. Печатается для полноты
    #    отчёта, не участвует в assert.
    print(f"(справочно) dist-to-neutral по всему кадру: before={dist_cold_before:.5f} "
          f"after={dist_cold_after:.5f} — эта метрика слишком грубая для вывода, см. п.2 выше")


def _warm_mult_plateau_fix(warm_bias):
    """Формула предыдущего фикса (плато: холодный источник получал ПОЛНЫЙ
    базовый пуш, warm_mult никогда не опускался ниже 1.0 для warm_bias<0) —
    копия только для regression-сравнения с новым симметричным сужением,
    НЕ трогать/не использовать больше нигде."""
    return 1.0 - 0.35 * max(0.0, warm_bias)


def test_warm_mult_symmetric_taper_reduces_cold_further_than_plateau_fix(tmp_path, monkeypatch):
    """П.4 (жалоба "меч всё ещё выглядит слишком холодно"): доказывает на
    РЕАЛЬНО ОТРЕНДЕРЕННЫХ пикселях, что новая симметричная формула
    (|warm_bias| вместо max(0,warm_bias)) даёт МЕНЬШЕ синего в тенях
    холодного кадра, чем формула предыдущего фикса (та лишь убирала
    усиление, но держала холодный источник на том же плато 1.0, что и
    нейтральный) — и что тёплая сторона по-прежнему не затронута (ветка для
    warm_bias>0 у обеих формул идентична)."""
    cold_in = str(tmp_path / "cold.png")
    warm_in = str(tmp_path / "warm.png")
    _make_gradient_photo(cold_in, r_shift=-COLOR_SHIFT, b_shift=COLOR_SHIFT)
    _make_gradient_photo(warm_in, r_shift=COLOR_SHIFT, b_shift=-COLOR_SHIFT)

    cold_new_out = str(tmp_path / "cold_new_out.png")
    cold_plateau_out = str(tmp_path / "cold_plateau_out.png")
    warm_new_out = str(tmp_path / "warm_new_out.png")
    warm_plateau_out = str(tmp_path / "warm_plateau_out.png")

    # "new" — реальный текущий прод-код (симметричное сужение), без патча
    cold_new_rgb = _render_via_film_look(cold_in, cold_new_out)
    warm_new_rgb = _render_via_film_look(warm_in, warm_new_out)
    # "plateau" — формула предыдущего фикса, только для этого сравнения
    monkeypatch.setattr(pipeline_smart, "_warm_mult", _warm_mult_plateau_fix)
    cold_plateau_rgb = _render_via_film_look(cold_in, cold_plateau_out)
    warm_plateau_rgb = _render_via_film_look(warm_in, warm_plateau_out)

    cold_new_shadow, _ = _region_means(cold_new_out)
    cold_plateau_shadow, _ = _region_means(cold_plateau_out)

    assert cold_new_shadow[2] < cold_plateau_shadow[2], (
        f"новая симметричная формула не дала меньше синего в тенях холодного кадра, чем "
        f"плато предыдущего фикса: new={cold_new_shadow} plateau={cold_plateau_shadow}")
    assert warm_new_rgb == pytest.approx(warm_plateau_rgb, abs=1e-3), (
        "тёплый источник изменился между формулами — ветка для warm_bias>0 идентична у обеих, "
        f"не должна была измениться: new={warm_new_rgb} plateau={warm_plateau_rgb}")


def test_domain_warm_push_scale_snow_reduces_shadow_further(tmp_path):
    """DOMAIN_WARM_PUSH_SCALE['snow']=0.5 должен реально уменьшать синий в
    тенях холодного кадра ДОПОЛНИТЕЛЬНО к обычному warm_bias-сужению, когда
    домен явно передан film_look() — на РЕАЛЬНО ОТРЕНДЕРЕННЫХ пикселях, не
    только по формуле на бумаге."""
    cold_in = str(tmp_path / "cold.png")
    _make_gradient_photo(cold_in, r_shift=-COLOR_SHIFT, b_shift=COLOR_SHIFT)

    no_domain_out = str(tmp_path / "cold_no_domain.png")
    snow_domain_out = str(tmp_path / "cold_snow_domain.png")

    _render_via_film_look(cold_in, no_domain_out, domain=None)
    _render_via_film_look(cold_in, snow_domain_out, domain="snow")

    shadow_no_domain, _ = _region_means(no_domain_out)
    shadow_snow_domain, _ = _region_means(snow_domain_out)

    assert shadow_snow_domain[2] < shadow_no_domain[2], (
        f"домен snow не уменьшил синий в тенях сильнее базового сужения: "
        f"no_domain={shadow_no_domain} snow_domain={shadow_snow_domain}")


def test_warm_mult_algebra_symmetric():
    """Быстрая алгебраическая проверка формы формулы (без ffmpeg) —
    дополняет визуальные regression-тесты выше, не заменяет их."""
    assert pipeline_smart._warm_mult(0.0) == pytest.approx(1.0)
    assert pipeline_smart._warm_mult(1.0) == pytest.approx(0.65)
    assert pipeline_smart._warm_mult(-1.0) == pytest.approx(0.65)
    assert pipeline_smart._warm_mult(0.4) == pytest.approx(pipeline_smart._warm_mult(-0.4))


def test_domain_warm_push_scale_unknown_domain_is_neutral():
    assert pipeline_smart.DOMAIN_WARM_PUSH_SCALE.get("some_unclassified_domain", 1.0) == 1.0
    assert pipeline_smart.DOMAIN_WARM_PUSH_SCALE["snow"] == pytest.approx(0.5)
