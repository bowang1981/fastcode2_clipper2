#include "rdtsc.h"
#if ! (defined(WIN32 ) || defined(WIN64))
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		RDTSC(t0);
		RDTSC(t1);
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
#else
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		__try {
		    RDTSC(t0);
		    RDTSC(t1);
		} __except ( 1) {
		    return 0;
		}
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
#endif
