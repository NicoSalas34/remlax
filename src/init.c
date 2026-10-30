/* Registration of the compiled code of remlax.
 *
 * The shared library holds one routine, rx_selinv_brut (selinv_c.c), the
 * kernel of the selected inversion of the mixed model equations engine. R
 * never calls it: the Python solver loads the library through ctypes, from the
 * path that R passes in REMLAX_SELINV_LIB. No routine is registered for .Call,
 * and dynamic symbol lookup from R is turned off.
 */
#include <R.h>
#include <R_ext/Rdynload.h>

void R_init_remlax(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
