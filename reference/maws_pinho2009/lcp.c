#include <stdio.h>
#include <stdlib.h>
#include "lcp.h"

/*==============================================================================
	Creates an lcp array. This code has been adapted from the original code
	in M. Douglas McIlroy's webpage http://www.cs.dartmouth.edu/~doug/sarray/
	which relies on the method published in

		T. Kasai, G. Lee, H. Arimura, S.Arikawa and K. Park,
		"Linear-time longest-common-prefix computation in suffix
		arrays and its applications", Proc 12th Annual Conference on
		Combinatorial Pattern Matching, Springer, LNCS 2089
		(2001) 181-192.

------------------------------------------------------------------------------*/

static INT lcpa(const INT *a, const char *s0, INT *lcp, INT n)
	{
	INT i, h;
	char *s = (char*) s0;
	void* xmalloc(INT n);
	INT *inv = (INT *)xmalloc(n*sizeof(INT));

	if(inv == 0)
		return 0;

	for(i = 0 ; i < n ; i++)
		inv[a[i]] = i;

	h = 0; /* visit in string order */
	for(i = 0 ; i < n-1 ; i++) /* omit last, least suff */
		{
		INT x = inv[i]; /* i, j, x, h as in intro */
		INT j = a[x-1];
		char *p1 = s + i + h;
		char *p0 = s + j + h;

		while(*p1 && *p0 && *p1++ == *p0++)
			h++;

		lcp[x] = h;
		if(h > 0)
			h--;
		}

	lcp[0] = 0; /* least suffix has no predecessor */
	free(inv);
	return 1;
	}

INT *BuildLcpArray(const INT *a, const char *s, INT n)
	{
	void* xmalloc(INT n);
	INT *lcp = (INT *)xmalloc(n*sizeof(INT));

	if(lcpa(a, s, lcp, n) == 0)
		{
		free(lcp);
		return 0;
		}

	return lcp;
	}

