all : yosystest
	./yosystest cpt.v

yosystest : yosys.cpp
	g++ -o yosystest -std=c++17 -stdlib=libc++ -D_YOSYS_ -I/opt/yosys/share/include/ -L/usr/local/lib/yosys/ yosys.cpp -L. -lyosys

