all: Makefile
	make -f Makefile
Makefile: 
	cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON 

clean:
	rm -rf CMakeCache.txt Makefile CMakeFiles/ 
tags:
	ctags -R --langmap=c++:+.txx --langmap=c++:+.cu $(GADGETRON_SOURCE) .

.PHONY: tags
