/*------------------------------------------------------------------------------

Copyright 2009 SPL-IEETA University of Aveiro, All Rights Reserved.

These programs are supplied free of charge for research purposes only,
and may not be sold or incorporated into any commercial product. There is
ABSOLUTELY NO WARRANTY of any sort, nor any undertaking that they are
fit for ANY PURPOSE WHATSOEVER. Use them at your own risk. If you do
happen to find a bug, or have modifications to suggest, please report
the same to Armando J. Pinho, ap@ua.pt. The copyright notice above
and this statement of conditions must remain an integral part of each
and every copy made of these files.

The code included in files sarray.* and lcp.* have been slightly
adapted from the original code from M. Douglas McIlroy's website
http://www.cs.dartmouth.edu/~doug/sarray/.

------------------------------------------------------------------------------*/

/*
 * Generates minimal absent words (MAWs) using suffix arrays and the method
 * explained in our BMC Bioinformatics paper.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <zlib.h>

#include "sarray.h"
#include "lcp.h"
#include "stack.h"
#include "int.h"

INT nSymbols = 4;
char *alphabet = "ACGT";
INT charMap[256]; /* Char codes to consecutive integers (0,...,nSymbols-1) */
char *g_data;
char ***maws = NULL;
INT *nMaws = NULL; /* Number of MAWs of a given size */
INT *maxMaws = NULL; /* Number of MAWs of a given size allocated */
int maxSize; /* Search for MAWs at most of maxSize */
INT largestMaximalRepeat = 0;
INT writeMaws = 0; /* By default, count only, don't write to stdout */

int complement(int c)
	{
	switch(c)
		{
		case 'A':
			return 'T';

		case 'C':
			return 'G';

		case 'G':
			return 'C';

		case 'T':
			return 'A';

		}

	return '\0';
	}

int cmp(const void* a, const void* b)
	{
	return strcmp(g_data + *(INT*)a, g_data + *(INT*)b);
	}

void mk_list(char* data, INT n, INT* array)
	{
	INT i;

	g_data = data;
	for(i =	0 ; i <= n ; i++)
		array[i] = i;

	qsort(array, n+1, sizeof(INT), cmp);
	}

int CmpChars(const void* a, const void* b)
	{
    return *(char *)a > *(char *)b;
	}

void* xmalloc(INT n)
	{
	void* p = calloc(n, 1);

	if(p == NULL)
		{
		fprintf(stderr, "ERROR: calloc() failed (requested %Ld bytes)\n",
		  (long long)n);
		exit(-1);
		}

	return p;
	}

void* xrealloc(void* p, INT n)
	{
	void* q = realloc(p, n);

	if(q == NULL)
		{
		fprintf(stderr, "ERROR: realloc() failed (requested %Ld bytes)\n",
		  (long long)n);
		exit(-1);
		}

	return q;
	}

void NewMaw(char *str, INT size, char l, char r)
	{
	if(nMaws[size] == maxMaws[size])
		{
		maxMaws[size] += 1 << 12;
		maws[size] = xrealloc(maws[size],
		  maxMaws[size] * sizeof(void *));
		}

	maws[size][nMaws[size]] = xmalloc(size+1);
	maws[size][nMaws[size]][0] = l;
	strncpy(maws[size][nMaws[size]]+1, str, size-2);
	maws[size][nMaws[size]][size-1] = r;
	maws[size][nMaws[size]][size] = 0;
	nMaws[size]++;
	}

INT GenMaws(LcpInterval *lcpi, INT *array, char *str)
	{
	INT i;
	INT l, r, LalphaSize = 0, found = 0;
	INT Lalpha[nSymbols + 1]; /* Need also to accommodate the undef symbol */
	INT Ralpha[nSymbols];
	INT lr[nSymbols][nSymbols];

	memset(Lalpha, 0, (nSymbols + 1) * sizeof(INT));
	memset(Ralpha, 0, nSymbols * sizeof(INT));
	memset(lr, 0, nSymbols * nSymbols * sizeof(INT));

	/*
	 * Build the L_\alpha and R_\alpha sets and the lr matrix.
	 */
	for(i = lcpi->lIdx ; i <= lcpi->rIdx ; i++)
		{
		if(array[i] && str[array[i]-1])
			Lalpha[charMap[(unsigned)str[array[i]-1]]] = 1;

		else
			Lalpha[nSymbols] = 1; /* Undefined char */

		if(str[array[i]+lcpi->lcp])
			Ralpha[charMap[(unsigned)str[array[i]+lcpi->lcp]]] = 1;

		if(array[i] && str[array[i]-1] && str[array[i]+lcpi->lcp])
			lr[charMap[(unsigned)str[array[i]-1]]]
			  [charMap[(unsigned)str[array[i]+lcpi->lcp]]] = 1;

		}

	for(i = 0 ; i <= nSymbols ; i++)
		LalphaSize += Lalpha[i];

	if(LalphaSize == 1) /* This node is not left-diverse */
		return 0;

	if(lcpi->lcp > largestMaximalRepeat)
		largestMaximalRepeat = lcpi->lcp;

	if(lcpi->lcp + 2 > maxSize)
		return 0;

	for(l = 0 ; l < nSymbols ; l++)
		{
		if(!Lalpha[l])
			continue;

		for(r = 0 ; r < nSymbols ; r++)
			{
			if(!Ralpha[r] || lr[l][r])
				continue;

			/* Store (or just count) a new MAW */
			if(writeMaws) /* store */
				NewMaw(str+array[lcpi->lIdx], lcpi->lcp+2, alphabet[l],
				  alphabet[r]);

			else /* count only */
				nMaws[lcpi->lcp+2]++;

			found++;
			}

		}

	return found;
	}

int main(int ac, char *av[])
	{
	char *str = NULL; /* the generalized string, null-terminated */
	INT nStrs = 0; /* number of strings in str */
	INT strLen = 0; /* a generalized strlen(s); see below */
	INT len = 0; /* number of non-delimiting symbols in string s */
	INT maxStrLen = 0; /* number of bytes allocated for the string */
	INT *array; /* the suffix array */
	INT *lcp; /* the lcp array */
	INT found = 0; /* number of MAWs found */
	INT n, i;
	INT c, size, sum = 0;
	INT lessMem = 0;
	INT rc = 0;
	INT nargs = 2;
	INT writeSize = 0;
	gzFile fp;

	StackOfLcpIntervals *stack = CreateStackOfLcpIntervals(100);
	LcpInterval lcpi;
	INT lIdx; /* lower index of a lcp-interval */
	INT nLcpi = 0; /* number of lcp-intervals */

	/* needed to measure CPU time */
	clock_t tic, tac, start, end;
	double cpu_time_used;

	start = clock();

	/* usage */
	if (ac < 3)
		{
		fprintf(stderr, "Usage: MinimalAbsentWords maxSize\n");
		fprintf(stderr, "                          [ -m (less mem, but slower) ]\n");
		fprintf(stderr, "                          [ -w (write to stdout) ]\n");
		fprintf(stderr, "                          [ -rc (reversed complement) ]\n");
		fprintf(stderr, "                          [ -n len (write only words of size len) ]\n");
		fprintf(stderr, "                          file [ file ... ]\n\n");
		return 1;
		}

	if(sscanf(av[1], "%d", &maxSize) != 1)
		{
		fprintf(stderr, "ERROR: failed to get a maximum word size...\n");
		return 1;
		}

	for(n = 1 ; n < ac ; n++)
		if(strcmp("-m", av[n]) == 0)
			{
			lessMem = 1;
			nargs++;
			break;
			}

	for(n = 1 ; n < ac ; n++)
		if(strcmp("-w", av[n]) == 0)
			{
			writeMaws = 1;
			nargs++;
			break;
			}

	for(n = 1 ; n < ac ; n++)
		if(strcmp("-rc", av[n]) == 0)
			{
			rc = 1;
			nargs++;
			break;
			}

	for(n = 1 ; n < ac ; n++)
		if(strcmp("-n", av[n]) == 0)
			{
			writeSize = atoi(av[n+1]);
			writeMaws = 1;
			nargs += 2;
			break;
			}

	fprintf(stderr, "using indexes with %d bits\n", (int)(sizeof(INT) * 8));

	for(n = 0 ; n < nSymbols ; n++)
		charMap[(unsigned)alphabet[n]] = n;

	fprintf(stderr, "reading data... ");
	tic = clock();

	for(i = nargs ; i < ac ; i++)
		{
		/* open the input file */
		if ((fp = gzopen(av[i], "r")) == NULL)
			{
			fprintf(stderr, "ERROR: failed to open file\n");
			return 1;
			}

		/* read the data leaving it in the null-terminated string s */
		while((c = gzgetc(fp)) != EOF)
			{
			c = toupper(c);

			if(c == '\n')
				continue;

			if(c == '>')
				{
				/* If not the first sequence, introduce a terminator */
				if(strLen)
					{
					if(strLen == maxStrLen)
						{
						maxStrLen += 1 << 20;
						str = xrealloc(str, maxStrLen);
						}

					str[strLen++] = 0;
					nStrs++;
					}

				while((c = gzgetc(fp)) != EOF && c != '\n')
					;

				continue;
				}

			if(strLen == maxStrLen)
				{
				maxStrLen += 1 << 20;
				str = xrealloc(str, maxStrLen);
				}

			if(strchr(alphabet, c))
				{
				str[strLen++] = c;
				len++;
				}

			else
				{
				str[strLen++] = 0; /* substitute delimiter by '\0' */
				nStrs++;
				}

			}

		if(str[strLen-1])
			{
			if(strLen == maxStrLen)
				{
				maxStrLen++;
				str = xrealloc(str, maxStrLen);
				}

			str[strLen++] = 0;
			nStrs++;
			}

		gzclose(fp);
		}

	if(rc == 1)
		{
		/* Compute the reversed complement */
		str = xrealloc(str, 2 * strLen);
		for(n = 0 ; n < strLen - 1 ; n++)
			str[strLen + n] = complement(str[strLen - n - 2]);

		str[strLen + n] = 0;

		strLen = 2 * strLen - 1;
		}

	else
		strLen--;

	tac = clock();
	cpu_time_used = ((double) (tac-tic)) / CLOCKS_PER_SEC;
	fprintf(stderr, "done: %g s\n", cpu_time_used);

	fprintf(stderr, "got %Ld valid DNA bases (i.e., ACGT or acgt)\n",
	  (long long)len);

	/* build the suffix array */
	array = xmalloc((strLen+1) * sizeof(INT));
	fprintf(stderr, "building suffix array... ");
	tic = clock();

	if(lessMem)
		mk_list(str, strLen, array);

	else
		bsarray((unsigned char*)str, array, strLen);

	tac = clock();
	cpu_time_used = ((double) (tac-tic)) / CLOCKS_PER_SEC;
	fprintf(stderr, "done: %g s\n", cpu_time_used);

	/* find the first non-empty string */
	n = 0;
	while(!*(str + array[n]))
		n++;

	fprintf(stderr, "found %Ld sub-string(s)\n", (long long)(n/2));

	/* build the lcp array */
	fprintf(stderr, "building lcp array... ");
	tic = clock();

	lcp = BuildLcpArray(array, str, strLen+1);

	tac = clock();
	cpu_time_used = ((double) (tac-tic)) / CLOCKS_PER_SEC;
	fprintf(stderr, "done: %g s\n", cpu_time_used);

	/* index 0 not used for better code readability */
	maws = xmalloc((maxSize+1) * sizeof(char **));
	nMaws = xmalloc((maxSize+1) * sizeof(INT));
	maxMaws = xmalloc((maxSize+1) * sizeof(INT));

	/* search for MAWs */
	fprintf(stderr, "searching minimal absent words... ");
	tic = clock();

	lcpi.lcp = 0;
	lcpi.lIdx = n;
	PushLcpInterval(stack, &lcpi);
	for(i = n+1 ; i <= strLen ; i++)
		{
		lIdx = i - 1;
		while(lcp[i] < TopLcpInterval(stack)->lcp)
			{
			lcpi = *PopLcpInterval(stack);
			lcpi.rIdx = i - 1;
			found += GenMaws(&lcpi, array, str);
			nLcpi++;
			lIdx = lcpi.lIdx;
			}

		if(lcp[i] > TopLcpInterval(stack)->lcp)
			{
			lcpi.lcp = lcp[i];
			lcpi.lIdx = lIdx;
			PushLcpInterval(stack, &lcpi);
			}

		}

	while(TopLcpInterval(stack)->lcp)
		{
		lcpi = *PopLcpInterval(stack);
		lcpi.rIdx = i - 1;
		found += GenMaws(&lcpi, array, str);
		nLcpi++;
		}

	tac = clock();
	cpu_time_used = ((double) (tac-tic)) / CLOCKS_PER_SEC;
	fprintf(stderr, "done: %g s\n", cpu_time_used);

	if(found)
		{
		for(size = 3 ; size <= maxSize ; size++)
			{
			sum += nMaws[size];
			if(nMaws[size])
				fprintf(stderr, "found %Ld minimal absent words of size %Ld\n",
				  (long long)nMaws[size], (long long)size);

			}

		if(writeMaws)
			for(size = 3 ; size <= maxSize ; size++)
				if(!writeSize || writeSize == size)
					for(n = 0 ; n < nMaws[size] ; n++)
						printf("%s\n", maws[size][n]);

		}

	fprintf(stderr, "size of largest maximal repeat: %Ld\n",
	  (long long)largestMaximalRepeat);
	fprintf(stderr, "found a total of %Ld minimal absent words\n",
	  (long long)found);
	end = clock();
	cpu_time_used = ((double) (end-start)) / CLOCKS_PER_SEC;
	fprintf(stderr, "total time used: %g s\n", cpu_time_used);
	return 0;
	}

