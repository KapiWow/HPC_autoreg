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

		Zeta<T> eps(size);
		
		#pragma omp parallel
		{
			//создание генераторов
        	int tid = omp_get_thread_num();
			std::string filename;
			filename = "3_task/mt_";
			filename += std::to_string(tid);
			std::ifstream fin(filename);
			mt_config conf;
			fin >> conf;
			parallel_mt generator(conf);
			std::normal_distribution<T> normal(T(0), std::sqrt(variance));
			//генерация волн
			#pragma omp for 
			for (int i = 0; i < size[0]; i++) {
				for (int j = 0; j < size[1]; j++) {
					for (int k = 0; k < size[2]; k++) {
						eps(i,j,k) = normal(generator);
					}
				}
			}
		}
		
		//проверка
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

		std::queue<Task> tasks;
		std::mutex mtx;
		std::condition_variable cv;
		int t_step = 40;
		int x_step = 8;
		int y_step = 8;
		int t_count = ((t1 + t_step - 1)/t_step);
		int x_count = ((x1 + x_step - 1)/x_step);
		int y_count = ((y1 + y_step - 1)/y_step);

		size3 matrixSize(t_count, x_count, y_count);
		Array3D<int> matrix(matrixSize);

		for (int i = 0; i < matrixSize[0]; i++) {
			for (int j = 0; j < matrixSize[1]; j++) {
				for (int k = 0; k < matrixSize[2]; k++) {
					int c = 3;
					if (i == 0)
						c -= 1;	
					if (j == 0)
						c -= 1;	
					if (k == 0)
						c -= 1;	
					//change matrix params
					if (c == 3)
						c = 7;
					else if (c == 2)
						c = 3;
					matrix(i,j,k) = c;

				}
			}
		}

		//create initial task
		Task task;
		task.t = 0;
		task.x = 0;
		task.y = 0;
		tasks.push(task);
		
		std::atomic<bool> stopped{false};
		#pragma omp parallel
		{
			std::unique_lock<std::mutex> lock{mtx};
			cv.wait(lock, [&] () {
				while (!tasks.empty()) {
					Task task = tasks.front();
					tasks.pop();
					int t = task.t;
					int x = task.x;
					int y = task.y;
					lock.unlock();
					
					int t_start = t*t_step;
					int t_end = std::min((t+1)*t_step, t1);
					int x_start = x*x_step;
					int x_end = std::min((x+1)*x_step, x1);
					int y_start = y*y_step;
					int y_end = std::min((y+1)*y_step, y1);
					
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

					if ((t==t_count-1)&&(x==x_count-1)&&(y==y_count-1)) {
						stopped = true;
						cv.notify_all();
					}

					lock.lock();
					for (size_t tn = 0; tn < newTasks.size(); tn++) {
						if ((newTasks[tn].t<t_count)&&
								(newTasks[tn].x<x_count)&&
								(newTasks[tn].y<y_count)) {
							matrix(newTasks[tn].t, newTasks[tn].x, newTasks[tn].y) -= 1;
							if (matrix(newTasks[tn].t, newTasks[tn].x, newTasks[tn].y) == 0) {
								tasks.push(newTasks[tn]);
								cv.notify_one();
							}
						}
					}
				}
				return stopped.load();
			});
		}
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
