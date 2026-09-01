#!/usr/bin/env bash
# Pilote de la grille : UNE cellule par processus, du plus petit au plus grand,
# avec un budget de temps. Les grandes cellules sont les plus informatives mais
# les plus couteuses : on les atteint seulement si le budget le permet, plutot
# que de perdre tout le balayage sur un walltime depasse.
set -u
RS="${RS:-Rscript}"; REPO="${REPO:-.}"; OUT="${OUT:-grille.csv}"
BK="${BK:-cpu}"; TAG="${TAG:-}"; BUDGET="${BUDGET:-18000}"
CELLS="${CELLS:-creux-dense dense-dense}"
NS="${NS:-2000 4000 8000 16000}"; QS="${QS:-500 2000}"; TS="${TS:-1 2 4 8}"
PLAFOND="${PLAFOND:-1800}"   # au-dela, on saute les cellules plus grandes encore
debut=$(date +%s)
for t in $TS; do for n in $NS; do for q in $QS; do
  [ "$q" -ge "$n" ] && continue          # q >= n : cas degenere, non identifie
  for cl in $CELLS; do
    ecoule=$(( $(date +%s) - debut ))
    if [ "$ecoule" -gt "$BUDGET" ]; then
      echo "[budget] $ecoule s ecoulees, arret propre avant $cl n=$n q=$q t=$t"
      exit 0
    fi
    cas="${cl%%-*}"; mot="${cl##*-}"
    t0=$(date +%s)
    timeout "$PLAFOND" $RS "$REPO/benchmarks/grille.R" --cas "$cas" --moteur "$mot" \
      --backend "$BK" --n "$n" --q "$q" --t "$t" --out "$OUT" --tag "$TAG" \
      --repo "$REPO" 2>&1 | grep -viE "codetools|adfun|remplie a 100"
    rc=$?
    [ "$rc" -eq 124 ] && echo "[plafond] $cl n=$n q=$q t=$t depasse $PLAFOND s, ignore"
    echo "    ($(( $(date +%s) - t0 )) s, cumul $(( $(date +%s) - debut )) s)"
  done
done; done; done
echo "[fin] $(( $(date +%s) - debut )) s au total"
