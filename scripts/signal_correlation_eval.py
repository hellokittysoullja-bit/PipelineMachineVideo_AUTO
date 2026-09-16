# -*- coding: utf-8 -*-
"""Есть ли ХОТЬ ОДИН бесплатный сигнал, коррелирующий с попаданием?

От этого зависят все три приоритета внешнего плана: фильтр памяти для
RAG, выбор из N для self-consistency и разметка пар для DPO. Если
корреляции нет ни у одного кандидата — два из трёх механизмов не на чем
строить, и это надо знать ДО, а не после месяца работы.
"""
import json, re, sys, os
# Замороженные прогоны 16.09 лежат в репозитории — замер воспроизводим
# без единого нового вызова модели и без ключей.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, 'docs', 'quality')
def _rows(name):
    with open(os.path.join(D, 'shot_brief_' + name + '.json'), encoding='utf-8') as f:
        return json.load(f)['arms']['C']['rows']
base, reas, q35 = _rows('baseline'), _rows('reasoning_on'), _rows('qwen35')

STOP = set("a an the of on in at with and or for to is are его её".split())
def words(s):
    return {w for w in re.findall(r"[a-z]{3,}", (s or '').lower()) if w not in STOP}

def rate(units, rows):
    """Доля попаданий среди набора юнитов."""
    hit = sum(1 for k in units if rows[k].get('subject_hit'))
    return hit, len(units), (100.0*hit/len(units) if units else 0)

both = [k for k in base if base[k].get('shot_en') and reas.get(k, {}).get('shot_en')]
print(f"юнитов, где ответили оба прогона: {len(both)}\n")

# --- СИГНАЛ 1: согласие двух независимых прогонов -------------------------
print("СИГНАЛ 1 — согласие двух прогонов (основа self-consistency и фильтра памяти)")
sims = []
for k in both:
    a, b = words(base[k]['shot_en']), words(reas[k]['shot_en'])
    j = len(a & b) / max(1, len(a | b))
    sims.append((j, k))
sims.sort(reverse=True)
half = len(sims)//2
agree = [k for _, k in sims[:half]]
disagree = [k for _, k in sims[half:]]
h1, n1, p1 = rate(agree, base); h2, n2, p2 = rate(disagree, base)
print(f"  верхняя половина по согласию:  {h1}/{n1} = {p1:.1f}% попаданий")
print(f"  нижняя половина по согласию:   {h2}/{n2} = {p2:.1f}% попаданий")
print(f"  РАЗДЕЛЕНИЕ: {p1-p2:+.1f} п.п.\n")

# --- СИГНАЛ 2: согласие ТРЁХ прогонов (третий — другая модель) ------------
tri = [k for k in both if q35.get(k, {}).get('shot_en')]
print(f"СИГНАЛ 2 — согласие ТРЁХ прогонов, третий на другой модели ({len(tri)} юнитов)")
sims3 = []
for k in tri:
    a, b, c = words(base[k]['shot_en']), words(reas[k]['shot_en']), words(q35[k]['shot_en'])
    j = (len(a&b)/max(1,len(a|b)) + len(a&c)/max(1,len(a|c)) + len(b&c)/max(1,len(b|c)))/3
    sims3.append((j, k))
sims3.sort(reverse=True)
h = len(sims3)//2
h1, n1, p1 = rate([k for _,k in sims3[:h]], base)
h2, n2, p2 = rate([k for _,k in sims3[h:]], base)
print(f"  верхняя половина: {h1}/{n1} = {p1:.1f}%   нижняя: {h2}/{n2} = {p2:.1f}%")
print(f"  РАЗДЕЛЕНИЕ: {p1-p2:+.1f} п.п.\n")

# --- СИГНАЛ 3: длина описания --------------------------------------------
print("СИГНАЛ 3 — длина описания в словах")
ln = sorted(((len(base[k]['shot_en'].split()), k) for k in base if base[k].get('shot_en')), reverse=True)
h = len(ln)//2
h1,n1,p1 = rate([k for _,k in ln[:h]], base); h2,n2,p2 = rate([k for _,k in ln[h:]], base)
print(f"  длинные: {h1}/{n1} = {p1:.1f}%   короткие: {h2}/{n2} = {p2:.1f}%")
print(f"  РАЗДЕЛЕНИЕ: {p1-p2:+.1f} п.п.\n")

# --- СИГНАЛ 4: заявленный тип кадра --------------------------------------
print("СИГНАЛ 4 — доля попаданий по типу кадра (object/scene/...)")
import collections
by = collections.defaultdict(list)
plan_path = None
for k in base:
    if base[k].get('shot_en'):
        by['все'].append(k)
for t in sorted(by):
    h,n,p = rate(by[t], base)
    print(f"  {t}: {h}/{n} = {p:.1f}%")
print()

# --- СИГНАЛ 5: era_ok (уже считается харнессом) --------------------------
print("СИГНАЛ 5 — era_ok (есть ли слово эпохи в описании)")
ok = [k for k in base if base[k].get('shot_en') and base[k].get('era_ok')]
no = [k for k in base if base[k].get('shot_en') and not base[k].get('era_ok')]
h1,n1,p1 = rate(ok, base); h2,n2,p2 = rate(no, base)
print(f"  со словом эпохи: {h1}/{n1} = {p1:.1f}%   без: {h2}/{n2} = {p2:.1f}%")
print(f"  РАЗДЕЛЕНИЕ: {p1-p2:+.1f} п.п.")
