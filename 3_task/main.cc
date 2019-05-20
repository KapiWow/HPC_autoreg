#include <fstream>
#include <string>
#include <chrono>

#include "parallel_mt.hh"

int main() {

	using namespace autoreg;

	parallel_mt_seq<521> seq(std::chrono::steady_clock::now().time_since_epoch().count());
	//a = seq();
	
	int mt_count = 32;
	for (int i = 0; i < mt_count; i++) {
		mt_config a;
		a = seq();
		std::string filename = "mt_";
		filename += std::to_string(i);
		std::ofstream fout(filename);
		fout << a;
	}
	//std::ofstream fout("mt");
	//fout << a;
	

	//parallel_mt mt(a);

//	mt();
}