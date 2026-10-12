<div align="center">

<img src="../../assets/banner.svg" alt="LLM Training Debugger" width="100%">

<br>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [한국어](README.ko.md) · [日本語](README.ja.md) · **Français**

<br>

[![tests](https://img.shields.io/badge/tests-584_passing-3fb950?style=flat-square)](../../tests)
[![python](https://img.shields.io/badge/python-3.9%2B-58a6ff?style=flat-square)](../../pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-8b949e?style=flat-square)](../../LICENSE)
[![status](https://img.shields.io/badge/status-alpha-d29922?style=flat-square)](#état-du-projet)
[![deepseek-harness](https://img.shields.io/badge/built_with-DeepSeek_Harness-4d6bfe?style=flat-square)](https://github.com/topics/deepseek-harness)

</div>

---

## Ce que c'est

Une **bibliothèque de surveillance pour l'entraînement PyTorch, fondée sur la preuve**. Elle
observe une boucle d'entraînement, transforme les anomalies de métriques en incidents et —
c'est le point qui compte — **refuse de répondre lorsque les preuves ne soutiennent pas de réponse**.

<div align="center">
<img src="../../assets/architecture.svg" alt="Architecture" width="100%">
</div>

La plupart des outils d'entraînement répondent à « que s'est-il passé ? ». Celui-ci pose une
question plus étroite et plus difficile :

> **Est-ce que je sais réellement — et sinon, qu'est-ce que j'ai exactement ?**

## Ce que ce n'est pas

Dit d'emblée, car une frontière honnête est plus utile qu'une liste de fonctionnalités :

- **Pas un remplacement pour les outils de suivi d'expériences.** Il écrit vers l'un d'eux ; ce
  n'est pas un tableau de bord.
- **Pas une analyse de cause racine.** Il rapporte *une étape et une métrique*. Il ne peut pas
  désigner un noyau, un tenseur ou une ligne de code.
- **Pas un contrôleur.** Il observe. Il ne coupera pas vos gradients, ne réinitialisera pas l'état
  de votre optimiseur, n'arrêtera pas votre entraînement.
- **Non éprouvé à grande échelle.** Chaque chiffre de ce dépôt a été mesuré sur un modèle de 4 à 6 couches.
- **Non validé.** Il n'existe pas d'échantillon de validation aveugle. `validated_capabilities` est
  vide par conception.

## Pourquoi il existe

Un entraînement produit une courbe de perte. Quand la courbe déraille, elle n'indique pas où, et
une réexécution reproduit rarement le problème.

Cette bibliothèque adopte la position suivante : **le produit n'est pas le graphe — le produit est
l'incident** : une affirmation délimitée, auditable et horodatée sur ce qui a été observé,
portant ses preuves et l'énoncé explicite de ce qui n'a pas pu être conclu.

Cette position a des conséquences dans toute la conception :

| Principe | Comment il apparaît dans le code |
|---|---|
| L'observation ne doit jamais casser l'entraînement | `emit()` ne lève jamais d'exception ; une file bornée P0–P3 abandonne plutôt que de bloquer |
| Une preuve doit être observée avant que son mécanisme soit affirmé | `OBSERVED` / `INFERRED` / `HYPOTHESIS` / `UNKNOWN` ; seul `OBSERVED` est promouvable |
| Une observation manquante n'est pas une observation négative | Chaque verdict porte `data_quality` et `completeness_warning` |
| Préférer l'abstention à une relation fabriquée | `INSUFFICIENT_EVIDENCE` est un verdict de premier ordre, pas une erreur |
| Certaines affirmations causales doivent être impossibles | `CAUSED_BY` et `ROOT_CAUSE_OF` **n'existent pas dans le schéma**, et un test garantit qu'ils n'existeront jamais |

## Prérequis

| | |
|---|---|
| **Python** | 3.9 – 3.13. Chaque version est réellement exercée en intégration continue. |
| **Cœur** | `numpy>=1.23`, `pyarrow>=12`, `duckdb>=0.10,<1.5` |
| **Adaptateur PyTorch** | `torch>=2.0` — facultatif, via l'extra `torch`. L'observateur, le stockage, les détecteurs, les épisodes, la fusion et la CLI n'importent pas torch. |
| **Développement** | `.[dev]` = `pytest` **et** torch, car la suite de tests l'importe. |

> **La borne supérieure sur `duckdb` est un contournement, pas une préférence.** duckdb 1.5.6
> casse `fetch_arrow_table()` avec `SystemError: <built-in function __import__> returned a
> result with an exception set`. Les 23 échecs en intégration continue se produisaient tous à
> cet unique appel ; les quelque 560 autres tests passaient et `import pyarrow` fonctionnait,
> donc la faute est dans le pont Arrow de duckdb et non dans pyarrow. Le défaut était
> invisible sous Python 3.9 parce que duckdb 1.5.x exige **Python >= 3.10** : 3.9 se résout
> en 1.4.5 et ne le rencontre jamais. **Quiconque utilise Python 3.10 ou plus récent avec une
> installation non épinglée recevait donc une version cassée.**
>
> Une tâche d'intégration continue nommée *latest dependencies (informational)* ignore
> délibérément toute borne supérieure et a le droit d'échouer. **Tant qu'elle est rouge, la
> borne reste nécessaire ; quand elle passera au vert, la borne devra être retirée.** Voir
> [KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md), point 15.

## Démarrage rapide

```bash
git clone https://github.com/YanKaFei/LLM-Training-Debugger.git
cd LLM-Training-Debugger
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Un exemple complet et exécutable :

```bash
python examples/quickstart.py
```

Ou intégrez-le à une boucle existante :

```python
from training_debugger.adapters.pytorch import PyTorchTrainingAdapter

adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="my-model")
adapter.run_start(steps=total_steps)

for step in range(1, total_steps + 1):
    with adapter.step(step) as tracker:
        loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        loss.backward()
        tracker.record_gradients()
        with tracker.phase("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tracker.record_parameters()

adapter.run_end()
```

## Ce que vous obtenez

<div align="center">
<img src="../../assets/verdicts.svg" alt="Verdicts" width="100%">
</div>

Chaque incident porte un `reason_code`, un `evidence_state`, une `promotion_disposition` et — si
le flux était incomplet — un avertissement `data_quality`. **Une exécution dont des événements ont
été perdus n'est jamais présentée comme une exécution saine.**

## Commandes

<div align="center">
<img src="../../assets/cli-demo.svg" alt="Démo CLI" width="100%">
</div>

```bash
training-debugger --root <store> runs                  # lister les exécutions
training-debugger --root <store> verdicts  <run>       # tous les verdicts, abstentions incluses
training-debugger --root <store> incidents <run>       # incidents avec leurs preuves
training-debugger --root <store> incident  <run> <id>  # un incident, en entier
training-debugger --root <store> metrics   <run> [m]   # interroger les lignes de métriques
training-debugger --root <store> diagnose  <run>       # diagnostic déterministe
training-debugger --root <store> compare   <a> <b>     # première étape où deux exécutions divergent
training-debugger --root <store> scan      <r1> ...    # lot : quelle exécution est l'aberration
```

### `compare` — localiser une divergence

Aligne les séries de métriques de deux exécutions, utilise **la région précoce où elles
s'accordent comme modèle nul**, et rapporte la première étape qui en sort et y reste. Le bruit
partagé s'annule dans la différence, ce qui permet une localisation impossible à un seuil
mono-exécution.

Mesuré sur un corpus apparié à étapes d'injection connues : **9/9 paires défectueuses localisées à
l'étape d'injection exacte, 3/3 paires saines correctement abstentionnistes.**

Les métriques d'horloge et de mémoire sont exclues par défaut — *une différence de temps n'est pas
une divergence algorithmique*.

### `scan` — trouver l'aberration dans un lot

Chaque paire est comparée une fois ; le score d'une exécution est le nombre de pairs dont elle
diverge. **Il exige des exécutions comparables** : fixez la graine et la configuration, ne variez
qu'un facteur. Sur un lot mélangeant les graines, il renvoie `NO_SINGLE_OUTLIER` et en explique la
raison, plutôt que de désigner une exécution.

## Capacités

Chaque capacité porte un niveau de preuve explicite. Rien ici n'est marqué comme validé.

| Capacité | Statut | Preuve |
|---|---|---|
| Observation des valeurs non finies (NaN / Inf) | `EXPERIMENTAL` | exacte ; sans seuil |
| Détection d'anomalies de gradient | `EXPERIMENTAL` | frontière de détection ×20 et ×40 → 6/6 ; ×2 et ×5 → 0/6 |
| Fusion d'incidents avec abstention | `EXPERIMENTAL` | précision 0.947 / rappel 0.500 (corpus de 42 exécutions) |
| Localisation de divergence deux-exécutions | `EXPERIMENTAL` | 9/9 exactes, 3/3 abstentions (corpus apparié) |
| Balayage d'aberrations par lot | `EXPERIMENTAL` | 5/5 pairs concordants sur un lot comparable |
| Santé d'activation par module | `EXPERIMENTAL` | module mort nommé et daté à OBSERVED 1.0 |
| Témoin d'appel d'optimiseur | `EXPERIMENTAL` | gelé ; désactivé par défaut |
| Inférence de relations (ordre seulement) | `UNSUPPORTED` | désactivé par défaut ; une tentative mesurée a fabriqué 16 associations |

## Limitations connues

Publiées parce qu'**un moniteur dont les échecs ne sont pas documentés ne peut pas être cru** :

- **Le rappel au niveau de l'exécution est de 0.500.** La moitié des défauts primaires du corpus de
  développement est manquée.
- **Incidents faux sur exécutions saines : 7.7 %**, contre un seuil auto-imposé de 5 %.
  **Ce seuil échoue.**
- **La surcharge de l'observateur échoue à son propre objectif de < 3 %** sur la charge de
  référence à 4 couches. Le coût est approximativement *fixe* par étape (0.2–0.9 ms), donc la
  fraction s'effondre quand la charge grandit : +11.0 % (4 couches) → +5.2 % (8) → +0.4 % (12).
  Au-delà d'une étape d'environ 17 ms, il est dans le budget. **Ceci est rapporté comme un échec,
  non excusé.**
- Aucun échantillon de validation aveugle n'existe ; tous les chiffres viennent du corpus de développement.
- Validé uniquement sur des modèles de 4 à 6 couches.

Liste complète : [KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md) · Journal de conception :
[DECISIONS.md](../DECISIONS.md) · État actuel : [DEVELOPMENT_STATE.md](../DEVELOPMENT_STATE.md)

## Reproduire les chiffres

```bash
pytest -q                                            # 584 réussis, 1 ignoré
python tests/experiments/full_pipeline_smoke.py      # 8/8
python examples/quickstart.py
python tests/performance/audit_observer_cost.py      # surcharge appariée, avec intervalles de confiance
```

Chaque chiffre clé de ce README est produit par un script dans `tests/experiments/` ou
`tests/performance/`, et la sortie brute est versionnée sous `outputs/`.

## État du projet

**Alpha.** Les interfaces ne sont pas stables. L'API peut changer sans préavis. Il est réellement
utile pour instrumenter une exécution et lire un verdict auditable ; ce n'est pas encore quelque
chose sur quoi bâtir un pipeline de production.

## Retours et discussions

**GitHub est le seul lieu de discussion de ce projet.** Il n'y a ni serveur de discussion, ni liste
de diffusion, ni autre canal — tout ce qui y est soulevé est donc visible par tous et reste
consultable.

| | |
|---|---|
| **Un bug ?** | [Ouvrir une issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **Une question, ou envie de discuter d'un choix de conception ?** | [Lancer une discussion](https://github.com/YanKaFei/LLM-Training-Debugger/discussions) |
| **Une fonctionnalité souhaitée ?** | [Ouvrir une issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **En désaccord avec une affirmation de ce document ?** | [Ouvrir une issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) — faites-le, s'il vous plaît |

**Le désaccord est bienvenu, et il l'est tout particulièrement à propos des chiffres.** Si vous
reproduisez une mesure de ce README et obtenez un résultat différent, c'est un rapport de bug qui
mérite d'être déposé. Si une affirmation ici n'est pas soutenue par les preuves de ce dépôt,
**cette affirmation est fausse** et doit être corrigée ou retirée.

Lors d'un rapport, veuillez inclure : la commande exécutée, la forme du modèle et le nombre
d'étapes, votre version de PyTorch, et la sortie complète.

## Contribuer

Voir [CONTRIBUTING.md](../../CONTRIBUTING.md). En bref : **les changements qui affaiblissent une
affirmation sont aussi bienvenus que ceux qui la renforcent**, et **un test ne doit jamais être
affaibli pour le faire passer**.

## Licence

[Apache-2.0](../../LICENSE).

<div align="center">
<br>
<sub>Construit avec <a href="https://github.com/topics/deepseek-harness">DeepSeek Harness</a>.</sub>
</div>
