#ifndef STACK_INCLUDED
#define STACK_INCLUDED

#include "int.h"

#define STACK_INCREMENT 10

typedef struct
	{
	INT lcp;
	INT lIdx;
	INT rIdx;
	}
LcpInterval;

typedef struct
	{
	INT sp;
	INT size;
	LcpInterval *ptr;
	}
StackOfLcpIntervals;

StackOfLcpIntervals *CreateStackOfLcpIntervals(INT);
LcpInterval *TopLcpInterval(StackOfLcpIntervals *s);
LcpInterval *PopLcpInterval(StackOfLcpIntervals *);
LcpInterval *PushLcpInterval(StackOfLcpIntervals *, LcpInterval *);
void ClearStackOfLcpIntervals(StackOfLcpIntervals *);
INT LengthOfStackOfLcpIntervals(StackOfLcpIntervals *);
void FreeStackOfLcpIntervals(StackOfLcpIntervals *);

#endif /* STACK_INCLUDED */

