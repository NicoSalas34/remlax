"""Finition d'un ajustement REML : Newton a region de confiance.

POURQUOI. Sur les gros modeles multivaries (mvD7 : 198 parametres), L-BFGS-B
passe des heures dans une longue traine ou il gagne quelques milliemes de
-2logL par centaine d'iterations. Deux proprietes du probleme l'expliquent :
des variances qui s'eteignent (en theta = log sigma, le gradient est
proportionnel a sigma^2 et s'annule a l'approche du plancher) et des
directions plates (covariances mal identifiees). Une methode du second ordre
y converge en quelques pas, a condition de traiter ces deux cas.

LE SCHEMA. A chaque pas, on resout le sous-probleme de region de confiance

    min_d  g'd + d'Hd / 2   sous   ||d|| <= Delta

exactement, par la decomposition propre de H restreint aux parametres libres
(More et Sorensen) : d(mu) = -V (V'g) / (w + mu), mu choisi pour que ||d|| =
Delta quand le pas de Newton sort de la region ou que H n'est pas definie
positive. Une direction plate ou de courbure negative est donc bornee par
Delta au lieu d'exploser. Le rapport gain reel / gain predit regle Delta.

LE HESSIEN coute 2p gradients (differences finies centrees). Il est calcule
au depart puis mis a jour par SR1 a chaque pas accepte (une evaluation par
pas) ; il n'est recalcule que si le modele quadratique cesse de predire les
gains (deux rapports mediocres de suite), et une derniere fois pour confirmer
la convergence.

LES VARIANCES QUI S'ETEIGNENT. Un parametre libre proche du plancher, que le
gradient pousse vers le bas, est plaque au plancher si -2logL ne monte pas de
plus de `tol_plancher` : on evite ainsi la descente asymptotique vers -12.
Si le gradient le repousse ensuite vers l'interieur, il redevient libre.

L'ARRET porte sur le decrement de Newton g'H^+g / 2 (gain de -2logL encore
disponible dans le modele local, sur les directions de courbure non
negligeable) et sur le gain des derniers pas, pas sur une tolerance relative
de l'objectif.
"""
import time

import numpy as np


def _libres(theta, g, lo, hi, tol=1e-10):
    bas = (theta <= lo + tol) & (g > 0)
    haut = (theta >= hi - tol) & (g < 0)
    return ~(bas | haut)


def _pas_rc(g, H, delta):
    """Pas de region de confiance exact (decomposition propre). Rend (d, pred)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    a = V.T @ g
    w_min = w.min()

    def d_de(mu):
        return -V @ (a / (w + mu))

    if w_min > 0:
        d = d_de(0.0)
        if np.linalg.norm(d) <= delta:
            return d, -(g @ d + 0.5 * d @ H @ d)
    # CAS DIFFICILE (More et Sorensen) : courbure negative et gradient presque
    # nul sur son vecteur propre. d(mu) ne peut alors pas atteindre le bord par
    # mu seul ; on complete par un deplacement le long de ce vecteur propre.
    i_min = int(np.argmin(w))
    if w_min < 0 and abs(a[i_min]) <= 1e-8 * max(np.linalg.norm(g), 1e-300):
        mu = -w_min
        a2 = a.copy(); a2[i_min] = 0.0
        ok = np.abs(w + mu) > 1e-14 * max(1.0, abs(w).max())
        d0 = -V[:, ok] @ (a2[ok] / (w[ok] + mu))
        reste = delta ** 2 - d0 @ d0
        if reste > 0:
            d = d0 + np.sqrt(reste) * V[:, i_min]
            return d, -(g @ d + 0.5 * d @ H @ d)
    # mu tel que ||d(mu)|| = delta, mu > max(0, -w_min) : bissection en echelle log
    m_lo = max(0.0, -w_min) + 1e-12 * max(1.0, abs(w).max())
    m_hi = m_lo + np.linalg.norm(g) / delta + abs(w).max()
    while np.linalg.norm(d_de(m_hi)) > delta:
        m_hi *= 2.0
    for _ in range(100):
        m = np.sqrt(m_lo * m_hi) if m_lo > 0 else 0.5 * (m_lo + m_hi)
        if np.linalg.norm(d_de(m)) > delta:
            m_lo = m
        else:
            m_hi = m
        if m_hi - m_lo <= 1e-10 * m_hi:
            break
    d = d_de(m_hi)
    return d, -(g @ d + 0.5 * d @ H @ d)


def decrement(g, H, rel=1e-8, plancher=1e-6):
    """Gain de -2logL encore disponible dans le modele local (decrement de Newton).

    REGULARISE, PAS TRONQUE. Une premiere version ignorait les directions de
    courbure inferieure a rel * max|w|. Quand une courbure est enorme (variance
    pres du plancher), ce seuil relatif ecartait des directions portant encore
    de vrais gains : sur un fa a 28 environnements (drops2_fa1), la finition
    concluait a -2logL 5598,78 quand l'optimum est a 5595,41. Ici chaque
    direction compte avec la courbure max(w, plancher) : une direction plate
    n'est negligee que si le gradient y est lui-meme negligeable. Une courbure
    negative significative rend +inf : ce n'est pas un minimum.
    """
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    a = V.T @ g
    echelle = max(abs(w).max(), 1e-300)
    if np.any((w < -rel * echelle) & (np.abs(a) > 1e-8 * max(np.linalg.norm(g), 1e-300))):
        return float("inf")
    return 0.5 * float(np.sum(a ** 2 / np.maximum(w, max(plancher, rel * echelle))))


def finir(fun_jac, theta, lo, hi, hessien, maxit=300, tol_dec=1e-6, tol_gain=1e-7,
          delta0=1.0, delta_max=10.0, fenetre_plancher=6.0, tol_plancher=1e-6,
          verbose=True, compteur=None):
    """Newton a region de confiance depuis theta.

    fun_jac(theta) -> (-2logL, gradient) en echelle NATURELLE ; hessien(theta)
    -> H (p x p). lo, hi : bornes (vecteurs). Rend (theta, info).
    """
    t0 = time.time()
    theta = np.clip(np.asarray(theta, dtype=np.float64), lo, hi)
    n_eval = [0]

    def evaluer(th):
        n_eval[0] += 1
        if compteur is not None:
            compteur["n_eval"] = compteur.get("n_eval", 0) + 1
        v, g = fun_jac(th)
        return float(v), np.asarray(g, dtype=np.float64)

    f, g = evaluer(theta)
    H = hessien(theta); n_hess = 1; frais = True
    delta = delta0
    mauvais, petits = 0, 0
    msg, ok = "maxit atteint", False

    def plaquer(theta, f, g):
        """Plaque au plancher les variances libres que le gradient pousse vers le bas."""
        cand = (theta > lo + 1e-10) & (theta < lo + fenetre_plancher) & (g > 0)
        if not cand.any():
            return theta, f, g, 0
        th2 = theta.copy(); th2[cand] = lo[cand]
        f2, g2 = evaluer(th2)
        if np.isfinite(f2) and f2 <= f + tol_plancher:
            return th2, f2, g2, int(cand.sum())
        return theta, f, g, 0

    theta, f, g, n_pl = plaquer(theta, f, g)
    if verbose:
        print("  [finition] depart -2logL %.6f | %d parametre(s) plaque(s) au plancher | Hessien %d gradients"
              % (f, n_pl, 2 * len(theta)), flush=True)
    for it in range(maxit):
        L = _libres(theta, g, lo, hi)
        gL, HL = g[L], H[np.ix_(L, L)]
        dec = decrement(gL, HL)
        if dec < tol_dec and petits >= 1:
            # avant de conclure : plaquer les variances qui s'eteignent (une
            # evaluation) ; si certaines le sont, le probleme a change de face
            theta, f, g, n_pl = plaquer(theta, f, g)
            if n_pl:
                if verbose:
                    print("  [finition] %d parametre(s) plaque(s) au plancher" % n_pl, flush=True)
                petits = 0
                continue
            if frais:
                msg, ok = "decrement %.2e < %.0e" % (dec, tol_dec), True
                break
            H = hessien(theta); n_hess += 1; frais = True        # confirmer avec un Hessien frais
            continue
        dL, pred = _pas_rc(gL, HL, delta)
        d = np.zeros_like(theta); d[L] = dL
        th_new = np.clip(theta + d, lo, hi)
        s = th_new - theta
        f_new, g_new = evaluer(th_new)
        gain = f - f_new if np.isfinite(f_new) else -np.inf
        rho = gain / pred if pred > 0 else -1.0
        if rho < 0.25:
            delta = max(0.25 * np.linalg.norm(s), 1e-8)
        elif rho > 0.75 and np.linalg.norm(dL) >= 0.99 * delta:
            delta = min(2.0 * delta, delta_max)
        if gain > 0:
            y = g_new - g
            # mise a jour SR1 de H (gardee si le denominateur n'est pas degenere)
            r = y - H @ s
            den = r @ s
            if abs(den) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
                H = H + np.outer(r, r) / den
                frais = False
            theta, f, g = th_new, f_new, g_new
            petits = petits + 1 if gain < tol_gain else 0
            mauvais = 0 if rho >= 0.25 else mauvais + 1
        else:
            mauvais += 1
        if verbose:
            print("  [finition] %3d | -2logL %.8f | gain %.2e | rho %6.2f | Delta %.2e | decrement %.2e | %d libres | %.0f s"
                  % (it + 1, f, gain, rho, delta, dec, int(L.sum()), time.time() - t0), flush=True)
        if mauvais >= 2:
            # le modele quadratique ne predit plus : Hessien frais, et plaquage
            theta, f, g, n_pl = plaquer(theta, f, g)
            H = hessien(theta); n_hess += 1; frais = True; mauvais = 0
            delta = max(delta, 1e-3)
            if verbose and n_pl:
                print("  [finition] %d parametre(s) plaque(s) au plancher" % n_pl, flush=True)
        if delta <= 1e-8 and frais:
            msg, ok = "region de confiance effondree (Delta %.1e)" % delta, dec < 1e-3
            break
    info = dict(success=ok, message="finition : %s" % msg, n_eval=n_eval[0], n_hessiens=n_hess,
                decrement=decrement(g[_libres(theta, g, lo, hi)], H[np.ix_(*(2 * [_libres(theta, g, lo, hi)]))]),
                secondes=time.time() - t0)
    if verbose:
        print("  [finition] fin : %s | -2logL %.8f | %d evaluations, %d Hessiens | %.0f s"
              % (msg, f, n_eval[0], n_hess, info["secondes"]), flush=True)
    return theta, info
