/* kernel/g6k_config.h.  Generated from g6k_config.h.in by configure.  */
/* kernel/g6k_config.h.in.  Generated from configure.ac by autoheader.  */

/* enable extended statistics */
/* #undef ENABLE_EXTENDED_STATS */

/* enable statistics */
/* #undef ENABLE_STATS */

/* major version */
#define G6K_MAJOR_VERSION 0

/* micro version */
#define G6K_MICRO_VERSION 2

/* minor version */
#define G6K_MINOR_VERSION 1

/* full version */
#define G6K_VERSION 0.1.2

/* long version string */
#define G6K_VERSION_INFO 

/* Define to 1 to support Advanced Bit Manipulation */
#define HAVE_ABM 1

/* Define to 1 to support Multi-Precision Add-Carry Instruction Extensions */
#define HAVE_ADX 1

/* Define to 1 to support Advanced Encryption Standard New Instruction Set
   (AES-NI) */
#define HAVE_AES 1

/* Support Altivec instructions */
/* #undef HAVE_ALTIVEC */

/* Define to 1 to support Advanced Vector Extensions */
#define HAVE_AVX 1

/* Define to 1 to support Advanced Vector Extensions 2 */
#define HAVE_AVX2 1

/* Define to 1 to support AVX-512 Byte and Word Instructions */
/* #undef HAVE_AVX512_BW */

/* Define to 1 to support AVX-512 Conflict Detection Instructions */
/* #undef HAVE_AVX512_CD */

/* Define to 1 to support AVX-512 Doubleword and Quadword Instructions */
/* #undef HAVE_AVX512_DQ */

/* Define to 1 to support AVX-512 Exponential & Reciprocal Instructions */
/* #undef HAVE_AVX512_ER */

/* Define to 1 to support AVX-512 Foundation Extensions */
/* #undef HAVE_AVX512_F */

/* Define to 1 to support AVX-512 Integer Fused Multiply Add Instructions */
/* #undef HAVE_AVX512_IFMA */

/* Define to 1 to support AVX-512 Conflict Prefetch Instructions */
/* #undef HAVE_AVX512_PF */

/* Define to 1 to support AVX-512 Vector Byte Manipulation Instructions */
/* #undef HAVE_AVX512_VBMI */

/* Define to 1 to support AVX-512 Vector Length Extensions */
/* #undef HAVE_AVX512_VL */

/* Define to 1 to support Bit Manipulation Instruction Set 1 */
#define HAVE_BMI1 1

/* Define to 1 to support Bit Manipulation Instruction Set 2 */
#define HAVE_BMI2 1

/* define if the compiler supports basic C++11 syntax */
#define HAVE_CXX11 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 to support Fused Multiply-Add Extensions 3 */
#define HAVE_FMA3 1

/* Define to 1 to support Fused Multiply-Add Extensions 4 */
/* #undef HAVE_FMA4 */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 to support Multimedia Extensions */
#define HAVE_MMX 1

/* Define to 1 to support Memory Protection Extensions */
#define HAVE_MPX 1

/* Define to 1 to support Prefetch Vector Data Into Caches WT1 */
/* #undef HAVE_PREFETCHWT1 */

/* Define if you have POSIX threads libraries and header files. */
#define HAVE_PTHREAD 1

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define to 1 to support Digital Random Number Generator */
#define HAVE_RDRND 1

/* Define to 1 to support Secure Hash Algorithm Extension */
/* #undef HAVE_SHA */

/* Define to 1 to support Streaming SIMD Extensions */
#define HAVE_SSE 1

/* Define to 1 to support Streaming SIMD Extensions */
#define HAVE_SSE2 1

/* Define to 1 to support Streaming SIMD Extensions 3 */
#define HAVE_SSE3 1

/* Define to 1 to support Streaming SIMD Extensions 4.1 */
#define HAVE_SSE4_1 1

/* Define to 1 to support Streaming SIMD Extensions 4.2 */
#define HAVE_SSE4_2 1

/* Define to 1 to support AMD Streaming SIMD Extensions 4a */
/* #undef HAVE_SSE4a */

/* Define to 1 to support Supplemental Streaming SIMD Extensions 3 */
#define HAVE_SSSE3 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Support VSX instructions */
/* #undef HAVE_VSX */

/* Define to 1 to support eXtended Operations Extensions */
/* #undef HAVE_XOP */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* maximum supported sieving dimension */
#define MAX_SIEVING_DIM 128

/* Name of package */
#define PACKAGE "g6k-kernel"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "g6k-kernel"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "g6k-kernel 0.1.2"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "g6k-kernel"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.1.2"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* enable templated dimensions */
/* #undef TEMPLATED_DIM */

/* Version number of package */
#define VERSION "0.1.2"

/* popcount hamming weight threshold for buckets */
#define XPC_BUCKET_THRESHOLD 102

/* popcount hamming weight threshold for pairs */
#define XPC_THRESHOLD 96
