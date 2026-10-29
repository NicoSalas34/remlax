# Structure residuelle
# @param struct "iid" | "diag" | "us"
# @param trait  facteur de caractere (longueur n) ; NULL = un seul caractere
# @param unit   identifiant d'unite : deux observations de la MEME unite sur
#               des caracteres differents sont correlees sous `us`.
# struct = "none" : AUCUNE residuelle (R = 0). V doit alors etre definie
# positive par les seuls termes - typiquement un terme de plante dont les
# incidences sont [I place sur le caractere t, W_t place sur le caractere t],
# structure us sur 2q colonnes, qui porte a la fois la residuelle, les effets
# environnementaux indirects (IEE) et leur covariance. Le moteur verifie V au
# point de depart et le dit en clair si elle ne l'est pas.
rx_residual <- function(struct = "iid", trait = NULL, unit = NULL, rank = 0L,
                        level = "id", order = 0L, coord = NULL, n_unit = NULL,
                        dims = NULL, opts = NULL, expr = NULL, sections = NULL,
                        rows = NULL, name = NULL) {
  struct <- match.arg(struct, c("iid", "diag", "us", "fa", "none"))
  if (struct == "none" && (!identical(level, "id") || !is.null(sections)))
    stop("rx_residual(\"none\") : ni structure entre unites ni sections, il n'y a pas de ",
         "residuelle a structurer.", call. = FALSE)
  level  <- match.arg(level, RX_LEVEL_STRUCTURES)
  structure(list(struct = struct, trait = trait, unit = unit, rank = as.integer(rank),
                 level = level, order = as.integer(order), coord = coord,
                 n_unit = n_unit, dims = dims, opts = opts, expr = expr,
                 sections = sections, rows = rows, name = name),
            class = "rx_residual")
}
