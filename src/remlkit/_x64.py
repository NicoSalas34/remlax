"""Double precision OBLIGATOIRE, activee avant toute creation de tableau JAX.

JAX est en float32 par defaut. Sur une vraisemblance REML c'est disqualifiant :
mesure sur ce paquet, un aller-retour Sigma -> theta -> Sigma perdait 2.6e-07 en
float32 contre ~1e-16 en float64, et le log-determinant d'une matrice mal
conditionnee derive bien plus vite encore. Ce module est importe en premier par
__init__.py : importer un sous-module de remlkit suffit donc a l'activer.
"""
import jax
jax.config.update("jax_enable_x64", True)
