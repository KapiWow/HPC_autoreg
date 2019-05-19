#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>             // for min, any_of, copy_n, for_each, generate
#include <cassert>               // for assert
#include <chrono>                // for duration, steady_clock, steady_clock...
#include <cmath>                 // for isnan
#include <cstdlib>               // for abs
#include <functional>            // for bind
#include <iostream>              // for operator<<, cerr, endl
#include <fstream>               // for ofstream
#include <random>                // for mt19937, normal_distribution
#include <stdexcept>             // for runtime_error
#include <vector>

#include <blitz/array.h>         // for Array, Range, shape, any

#include "sysv.hh"               // for sysv
#include "types.hh"              // for size3, ACF, AR_coefs, Zeta, Array2D
#include "voodoo.hh"             // for generate_AC_matrix

#include <condition_variable>
#include <mutex>   
#include <queue>
#include <atomic>
#include <thread>
#include "parallel_mt.hh"
#include <string>
#include <omp.h>

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template<class T>
	ACF<T>
	approx_acf(T alpha, T beta, T gamm, const Vec3<T>& delta, const size3& acf_size) {
		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm
			* blitz::exp(-alpha * (t*delta[0] + x*delta[1] + y*delta[2]))
//	 		* blitz::cos(beta * (t*delta[0] + x*delta[1] + y*delta[2]));
	 		* blitz::cos(beta * t * delta[0])
	 		* blitz::cos(beta * x * delta[1])
	 		* blitz::cos(beta * y * delta[2]);
		return acf;
	}

	template<class T>
	T white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		return acf(0,0,0) - blitz::sum(ar_coefs * acf);
	}

	template<class T>
	T ACF_variance(const ACF<T>& acf) {
		return acf(0,0,0);
	}

	/// Удаление участков разгона из реализации.
	template<class T>
	Zeta<T>
	trim_zeta(const Zeta<T>& zeta2, const size3& zsize) {
		using blitz::Range;
		using blitz::toEnd;
		size3 zsize2 = zeta2.shape();
		return zeta2(
			Range(zsize2(0) - zsize(0), toEnd),
			Range(zsize2(1) - zsize(1), toEnd),
			Range(zsize2(2) - zsize(2), toEnd)
		);
	}

	template<class T>
	bool is_stationary(AR_coefs<T>& phi) {
		return !blitz::any(blitz::abs(phi) > T(1));
	}

	template<class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf) {
		using blitz::Range;
		using blitz::toEnd;
		const int m = acf.numElements()-1;
		Array2D<T> acm = generate_AC_matrix(acf);
		//{ std::ofstream out("acm"); out << acm; }

		/**
		eliminate the first equation and move the first column of the remaining
		matrix to the right-hand side of the system
		*/
		Array1D<T> rhs(m);
		rhs = acm(Range(1, toEnd), 0);
		//{ std::ofstream out("rhs"); out << rhs; }

		// lhs is the autocovariance matrix without first
		// column and row
		Array2D<T> lhs(blitz::shape(m,m));
		lhs = acm(Range(1, toEnd), Range(1, toEnd));
		//{ std::ofstream out("lhs"); out << lhs; }

		assert(lhs.extent(0) == m);
		assert(lhs.extent(1) == m);
		assert(rhs.extent(0) == m);
		sysv<T>('U', m, 1, lhs.data(), m, rhs.data(), m);
		AR_coefs<T> phi(acf.shape());
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0,0,0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data()+1);
		//{ std::ofstream out("ar_coefs"); out << phi; }
		if (!is_stationary(phi)) {
			std::cerr << "phi.shape() = " << phi.shape() << std::endl;
			std::for_each(
				phi.begin(),
				phi.end(),
				[] (T val) {
					if (std::abs(val) > T(1)) {
						std::cerr << val << std::endl;
					}
				}
			);
			throw std::runtime_error("AR process is not stationary, i.e. |phi| > 1");
		}
		return phi;
	}

	template<class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

	/// Генерация белого шума по алгоритму Вихря Мерсенна и
	/// преобразование его к нормальному распределению по алгоритму Бокса-Мюллера.
	template<class T>
	Zeta<T>
	generate_white_noise(const size3& size, const T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// инициализация генератора
		//std::mt19937 generator;
		//#if !defined(DISABLE_RANDOM_SEED)
		//generator.seed(std::chrono::steady_clock::now().time_since_epoch().count());
		//#endif
		

		std::vector<parallel_mt> generators;
		std::vector<std::normal_distribution<T>> normals;
		int generator_count = 8;

		for (int i = 0; i < generator_count; i++) {
			std::string filename;
			filename = "3_tast/mt_";
			filename += std::to_string(i);
			std::ifstream fin(filename);
			mt_config conf;
			fin >> conf;
			parallel_mt generator(conf);
			generators.push_back(generator);		
			
			std::normal_distribution<T> normal(T(0), std::sqrt(variance));
			normals.push_back(normal);
		}	






		// генерация и проверка
		Zeta<T> eps(size);
		//std::mutex
		
		//std::cout<<size[2]<<std::endl;
		#pragma omp parallel for 
		for (int i = 0; i < size[0]; i++) {
        	int tid = omp_get_thread_num();
			std::normal_distribution<T> normal(T(0), std::sqrt(variance));
			//
			//#pragma omp critical
			
				//parallel_mt generator(generators[0]);
			parallel_mt generator(generators[tid]);
				//generator = generators[tid];
			
			
			//std::ifstream fin("3_tast/mt_0");
			//mt_config conf;
			//fin >> conf;
			//parallel_mt generator(generators[tid]);
			for (int j = 0; j < size[1]; j++) {
				for (int k = 0; k < size[2]; k++) {
					//if (tid == 0)
					//	std::cout<< "12312312"<<std::endl;
					eps(i,j,k) = normal(generator);
					//eps(i,j,k) = normal(generators[tid]);
					//eps(j,i,k) = normal(generators[tid]);
				}
			}
		}

		//std::normal_distribution<T> normal(T(0), std::sqrt(variance));
		//std::generate(std::begin(eps), std::end(eps), std::bind(normal, generator));
		//std::generate(std::begin(eps), std::end(eps), std::bind(normal, generator[0]));
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isnan<T>)) {
			throw std::runtime_error("white noise generator produced some NaNs");
		}
		return eps;
	}

	/// Генерация отдельных частей реализации волновой поверхности.
	template<class T>
	void generate_zeta(const AR_coefs<T>& phi, Zeta<T>& zeta) {
		const size3 fsize = phi.shape();
		const size3 zsize = zeta.shape();
		std::cout << fsize << std::endl;
		std::cout << zsize << std::endl;
		const int t1 = zsize[0];
		const int x1 = zsize[1];
		const int y1 = zsize[2];
		struct Task 
		{
			int t;
			int x;
			int y;
		};


		//std::vector<Task> tasks;
		//for (int s=0; s<=(t1+x1+y1-3); s++) {
		////for (int s=0; s<=3; s++) {
		//	int count = s;
		//	const int start_t = std::max(0, count-x1-y1-2);
		//	const int end_t = std::min(count, t1-1);
		//	//std::cout << s << std::endl;
		//	for (int t=start_t; t<=end_t; t++) {
		//		count = s-t;
		//		const int start_x = std::max(0,count-y1-1);
		//		const int end_x = std::min(count,y1-1);
		//		for (int x=start_x; x<=end_x; x++) {
		//			//count -= x;
		//			count = s-t-x;
		//			Task task;
		//			task.t = t;
		//			task.x = x;
		//			task.y = count;
		//			tasks.push_back(task);
		//			//std::cout << t << " : " << x << " : " << count << std::endl;
		//		}
		//	}
		//	#pragma omp parallel for
		//	for (size_t ii=0; ii<tasks.size(); ii++) {
		//		int t = tasks[ii].t;
		//		int x = tasks[ii].x;
		//		int y = tasks[ii].y;
		//		//if (t > 1390)
		//		//	std::cout << t << " : " << x << " : " << y << std::endl;
		//		const int m1 = std::min(t+1, fsize[0]);
		//		const int m2 = std::min(x+1, fsize[1]);
		//		const int m3 = std::min(y+1, fsize[2]);
		//		T sum = 0;
		//		for (int k=0; k<m1; k++)
		//			for (int i=0; i<m2; i++)
		//				for (int j=0; j<m3; j++)
		//					sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
		//		#pragma omp atomic
		//		zeta(t, x, y) += sum;
		//	}
		//	tasks.clear();
		//}




		//=========================================================================
		//

		std::queue<Task> tasks;
		std::vector<std::vector<std::vector<int>>> matrix;
		std::mutex mtx;
		//std::mutex mtxTable;
		std::condition_variable cv;
		size_t t_step = 40;
		size_t x_step = 8;
		size_t y_step = 8;
		//size_t t_step = 50;
		//size_t x_step = 8;
		//size_t y_step = 8;


		for (int i = 0; i < t1/t_step; i++) {
			std::vector<std::vector<int>> a;
			for (int j = 0; j < x1/x_step; j++) {
				std::vector<int> b;
				for (int k = 0; k < y1/y_step; k++) {
					int c = 3;
					if (i == 0)
						c -= 1;	
					if (j == 0)
						c -= 1;	
					if (k == 0)
						c -= 1;	
					
					
					if (c==3)
						c = 7;
					else if (c==2)
						c = 3;

					b.push_back(c);
				}
				a.push_back(b);
			}	
			matrix.push_back(a);
		}

		std::cout << matrix[0][0][0] << std::endl;
		Task task;
		task.t = 0;
		task.x = 0;
		task.y = 0;
		tasks.push(task);
		std::cout << t1 << "  " << x1 << "   " << y1 << std::endl;
		
		std::atomic<bool> stopped{false};
		#pragma omp parallel
		{
		std::unique_lock<std::mutex> lock{mtx};
		cv.wait(lock, [&stopped, &matrix, &tasks, &lock, &zeta, &phi, &fsize, &t1, &x1, &y1, &cv,
		&t_step, &x_step, &y_step] () {
			while(!tasks.empty()) {
				Task task = tasks.front();
				tasks.pop();
				//std::unlock_guard unlock{mtx};
				//lock.unlock();
				
				int t = task.t;
				int x = task.x;
				int y = task.y;
				//std::cout << t <<"   "<< x <<"   "<< y << std::endl;
				//std::cout << t + x + y << std::endl;

				int t_start = t*t_step;
				int t_end = std::min((t+1)*t_step, (size_t)t1);
				//std::cout << " t " << t_start <<"   "<< t_end << std::endl;
				
				int x_start = x*x_step;
				int x_end = std::min((x+1)*x_step, (size_t)x1);
				//std::cout << " x " << x_start <<"   "<< x_end << std::endl;
				
				int y_start = y*y_step;
				int y_end = std::min((y+1)*y_step, (size_t)y1);
				//std::cout << " y " << y_start <<"   "<< y_end << std::endl;
				lock.unlock();
				
				for (int t = t_start; t < t_end; t++)
					for (int x = x_start; x < x_end; x++)
						for (int y = y_start; y < y_end; y++) {
							const int m1 = std::min(t+1, fsize[0]);
							const int m2 = std::min(x+1, fsize[1]);
							const int m3 = std::min(y+1, fsize[2]);
							T sum = 0;
							for (int k=0; k<m1; k++)
								for (int i=0; i<m2; i++)
									for (int j=0; j<m3; j++)
										sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
						#pragma omp atomic
							zeta(t, x, y) += sum;
						}
				
				
				
				std::vector<Task> newTasks;
				newTasks.push_back(Task{t+1, x, y});
				newTasks.push_back(Task{t, x+1, y});
				newTasks.push_back(Task{t, x, y+1});
				newTasks.push_back(Task{t+1, x+1, y});
				newTasks.push_back(Task{t+1, x, y+1});
				newTasks.push_back(Task{t, x+1, y+1});
				newTasks.push_back(Task{t+1, x+1, y+1});
				lock.lock();
				for (size_t tn = 0; tn < newTasks.size(); tn++) {
					if ((newTasks[tn].t<t1/t_step)&&
							(newTasks[tn].x<x1/x_step)&&
							(newTasks[tn].y<y1/y_step)) {
						matrix[newTasks[tn].t][newTasks[tn].x][newTasks[tn].y] -= 1;
						if (matrix[newTasks[tn].t][newTasks[tn].x][newTasks[tn].y] == 0) {
							tasks.push(newTasks[tn]);
							cv.notify_one();
						}
					}
				}
				if ((t==t1/t_step -1)&&(x==x1/x_step-1)&&(y==y1/y_step-1)) {
					stopped = true;
					//std::cout << "stopped" << std::endl;
					cv.notify_all();
				}
				//lock.unlock();

			}
			return stopped.load();
			//bool res = stopped;
			//return res;
		});
		}
		
		//===================================================================

		//std::vector<Task> tasks;
		//
		//
		//for (int s=0; s<=(t1+x1+y1-3); s++) {
		//	int count = s;
		//	const int start_t = std::max(0, count-x1-y1-2);
		//	const int end_t = std::min(count, t1-1);
		//	//std::cout << s << std::endl;
		//	for (int t=start_t; t<=end_t; t++) {
		//		count = s-t;
		//		const int start_x = std::max(0,count-y1-1);
		//		const int end_x = std::min(count,y1-1);
		//		for (int x=start_x; x<=end_x; x++) {
		//			//count -= x;
		//			count = s-t-x;
		//			Task task;
		//			task.t = t;
		//			task.x = x;
		//			task.y = count;
		//			tasks.push_back(task);
		//			//std::cout << t << " : " << x << " : " << count << std::endl;
		//		}
		//	}
		//	#pragma omp parallel for
		//	for (size_t ii=0; ii<tasks.size(); ii++) {
		//		int t = tasks[ii].t;
		//		int x = tasks[ii].x;
		//		int y = tasks[ii].y;
		//		//if (t > 1390)
		//		//	std::cout << t << " : " << x << " : " << y << std::endl;
		//		const int m1 = std::min(t+1, fsize[0]);
		//		const int m2 = std::min(x+1, fsize[1]);
		//		const int m3 = std::min(y+1, fsize[2]);
		//		T sum = 0;
		//		for (int k=0; k<m1; k++)
		//			for (int i=0; i<m2; i++)
		//				for (int j=0; j<m3; j++)
		//					sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
		//		#pragma omp atomic
		//		zeta(t, x, y) += sum;
		//	}
		//	tasks.clear();
		//}




		//for (int t=0; t<t1; t++) {
		//	for (int x=0; x<x1; x++) {
		//		for (int y=0; y<y1; y++) {
		//			const int m1 = std::min(t+1, fsize[0]);
		//			const int m2 = std::min(x+1, fsize[1]);
		//			const int m3 = std::min(y+1, fsize[2]);
		//			T sum = 0;
		//			for (int k=0; k<m1; k++)
		//				for (int i=0; i<m2; i++)
		//					for (int j=0; j<m3; j++)
		//						sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
		//			zeta(t, x, y) += sum;
		//		}
		//	}
		//}
	}

	template<class T, int N>
	T mean(const blitz::Array<T,N>& rhs) {
		return blitz::sum(rhs) / rhs.numElements();
	}

	template<class T, int N>
	T variance(const blitz::Array<T,N>& rhs) {
		assert(rhs.numElements() > 0);
		const T m = mean(rhs);
		return blitz::sum(blitz::pow(rhs-m, 2)) / (rhs.numElements() - 1);
	}

}

#endif // AUTOREG_HH
