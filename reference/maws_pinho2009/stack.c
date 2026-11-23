#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

/*==============================================================================
	Creates a stack of lcp intervals.
------------------------------------------------------------------------------*/

StackOfLcpIntervals *CreateStackOfLcpIntervals(INT initialStackSize)

	{
	StackOfLcpIntervals *s;

	if(!(s = (StackOfLcpIntervals *)calloc(1, sizeof(StackOfLcpIntervals))))
		{
		fprintf(stderr,
		  "Error (CreateStackOfLcpIntervals): unable to allocate memory.\n");
		exit(1);
		}

	if(!(s->ptr = (LcpInterval *)calloc(initialStackSize, sizeof(LcpInterval))))
		{
		fprintf(stderr,
		  "Error (CreateStackOfLcpIntervals): unable to allocate memory.\n");
		exit(1);
		}

	s->size = initialStackSize;
	s->sp = -1;
	return s;
	}

/*==============================================================================
	Return a pointer to the element at the top of the stack.
	If the stack is empty, gives a fatal error.
------------------------------------------------------------------------------*/

LcpInterval *TopLcpInterval(StackOfLcpIntervals *s)

	{
	if(s->sp == -1)
		exit(1);

	return &(s->ptr[s->sp]);
	}

/*==============================================================================
	Pops an element of the stack.
------------------------------------------------------------------------------*/

LcpInterval *PopLcpInterval(StackOfLcpIntervals *s)

	{
	if(s->sp == -1)
		return(NULL);

	return &(s->ptr[s->sp--]);
	}

/*==============================================================================
	Pushes an element into the stack.
------------------------------------------------------------------------------*/

LcpInterval *PushLcpInterval(StackOfLcpIntervals *s, LcpInterval *ptr)

	{
	if(++s->sp >= s->size)
		{
		if(!(s->ptr = (LcpInterval *)realloc((void *)s->ptr,
		  (s->size + STACK_INCREMENT) * sizeof(LcpInterval))))
			{
			fprintf(stderr,
			  "Error (PushLcpInterval): unable to allocate memory.\n");
			exit(1);
			}

		s->size += STACK_INCREMENT;
		}

	s->ptr[s->sp] = *ptr;
	return ptr;
	}

/*==============================================================================
	Clears the stack.
------------------------------------------------------------------------------*/

void ClearStackOfLcpIntervals(StackOfLcpIntervals *s)

	{
	s->sp = -1;
	}

/*==============================================================================
	Returns the number of elements in the stack.
------------------------------------------------------------------------------*/

INT LengthOfStackOfLcpIntervals(StackOfLcpIntervals *s)

	{
	return s->sp + 1;
	}

/*==============================================================================
	Frees the memory allocated to the stack.
------------------------------------------------------------------------------*/

void FreeStackOfLcpIntervals(StackOfLcpIntervals *s)

	{
	free((void *)s->ptr);
	free((void *)s);
	}

