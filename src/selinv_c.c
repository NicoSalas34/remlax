/* Supernodal selected inversion (block Takahashi recurrence), the C kernel of
 * the mixed model equations engine. R compiles it at installation into the
 * package library; the Python solver (remlax/selinv.py) loads that library
 * through ctypes, from the path R passes in REMLAX_SELINV_LIB.
 *
 * No link dependency: the BLAS and LAPACK routines (dgemm, dtrsm, dpotri)
 * arrive as arguments, as the function pointers exposed by
 * scipy.linalg.cython_blas and cython_lapack.
 *
 * Storage (that of supernodal CHOLMOD):
 *   sup[ns+1]  first column of each supernode (0-based)
 *   pi[ns+1]   start of the rows of each supernode in s
 *   off[ns+1]  start of the block of each supernode in x and Z
 *   s          row indices; those of a supernode sorted, starting with its
 *              own columns
 *   sn_of[m]   supernode of each column
 *   x          Cholesky factor L, nrow x ncol blocks in column order
 *   Z          output: (L L')^-1 on the same storage
 *
 * Z_RJ = - Z_RR L_RJ L_JJ^-1 ;  Z_JJ = (L_JJ L_JJ')^-1 - (L_RJ L_JJ^-1)' Z_RJ.
 */
#include <stdlib.h>
#include <string.h>

typedef void (*dgemm_t)(char *, char *, int *, int *, int *, double *, double *, int *,
                        double *, int *, double *, double *, int *);
typedef void (*dtrsm_t)(char *, char *, char *, char *, int *, int *, double *, double *,
                        int *, double *, int *);
typedef void (*dpotri_t)(char *, int *, double *, int *, int *);

int rx_selinv_brut(int ns, const long long *off, const int *sup, const int *pi,
                   const int *s, const int *sn_of, const double *x, double *Z,
                   dgemm_t dgemm, dtrsm_t dtrsm, dpotri_t dpotri)
{
    int maxr = 0, maxc = 0;
    for (int J = 0; J < ns; J++) {
        int nr = pi[J + 1] - pi[J], nc = sup[J + 1] - sup[J];
        if (nr > maxr) maxr = nr;
        if (nc > maxc) maxc = nc;
    }
    double *Ljj = malloc(sizeof(double) * (size_t) maxc * maxc);
    double *Lt = malloc(sizeof(double) * (size_t) maxr * maxc);
    double *Zrr = malloc(sizeof(double) * (size_t) maxr * maxr);
    double *Zjj = malloc(sizeof(double) * (size_t) maxc * maxc);
    double *ZRJ = malloc(sizeof(double) * (size_t) maxr * maxc);
    if (!Ljj || !Lt || !Zrr || !Zjj || !ZRJ) {
        free(Ljj); free(Lt); free(Zrr); free(Zjj); free(ZRJ);
        return -1;
    }
    double one = 1.0, mone = -1.0, zero = 0.0;
    char cL = 'L', cR = 'R', cN = 'N', cT = 'T';
    int info = 0, statut = 0;

    for (int J = ns - 1; J >= 0 && statut == 0; J--) {
        int nJ = sup[J + 1] - sup[J], nr = pi[J + 1] - pi[J], nR = nr - nJ;
        const double *LJ = x + off[J];
        double *ZJ = Z + off[J];
        const int *rJ = s + pi[J];
        for (int j = 0; j < nJ; j++)
            for (int i = 0; i < nJ; i++)
                Ljj[i + (size_t) j * nJ] = (i >= j) ? LJ[i + (size_t) j * nr] : 0.0;
        memcpy(Zjj, Ljj, sizeof(double) * (size_t) nJ * nJ);
        dpotri(&cL, &nJ, Zjj, &nJ, &info);
        if (info != 0) { statut = 1000000 + J; break; }
        for (int j = 0; j < nJ; j++)
            for (int i = 0; i < j; i++) Zjj[i + (size_t) j * nJ] = Zjj[j + (size_t) i * nJ];
        if (nR > 0) {
            for (int j = 0; j < nJ; j++)
                for (int i = 0; i < nR; i++)
                    Lt[i + (size_t) j * nR] = LJ[nJ + i + (size_t) j * nr];
            dtrsm(&cR, &cL, &cN, &cN, &nR, &nJ, &one, Ljj, &nJ, Lt, &nR);   /* Lt = L_RJ Ljj^-1 */
            const int *R = rJ + nJ;
            for (int b = 0; b < nR && statut == 0; b++) {
                int c = R[b], K = sn_of[c];
                int cl = c - sup[K], nrK = pi[K + 1] - pi[K];
                const int *sK = s + pi[K];
                const double *ZK = Z + off[K] + (size_t) cl * nrK;
                int q = cl;
                for (int a = b; a < nR; a++) {
                    while (q < nrK && sK[q] < R[a]) q++;
                    if (q >= nrK || sK[q] != R[a]) { statut = 2000000 + J; break; }
                    Zrr[a + (size_t) b * nR] = ZK[q];
                }
            }
            if (statut) break;
            for (int b = 0; b < nR; b++)
                for (int a = 0; a < b; a++) Zrr[a + (size_t) b * nR] = Zrr[b + (size_t) a * nR];
            dgemm(&cN, &cN, &nR, &nJ, &nR, &mone, Zrr, &nR, Lt, &nR, &zero, ZRJ, &nR);
            dgemm(&cT, &cN, &nJ, &nJ, &nR, &mone, Lt, &nR, ZRJ, &nR, &one, Zjj, &nJ);
            for (int j = 0; j < nJ; j++)
                for (int i = 0; i < nR; i++)
                    ZJ[nJ + i + (size_t) j * nr] = ZRJ[i + (size_t) j * nR];
        }
        for (int j = 0; j < nJ; j++)
            for (int i = 0; i < nJ; i++)
                ZJ[i + (size_t) j * nr] = Zjj[i + (size_t) j * nJ];
    }
    free(Ljj); free(Lt); free(Zrr); free(Zjj); free(ZRJ);
    return statut;
}
