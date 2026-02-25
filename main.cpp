#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>

extern "C" {
		#include "xdrfile/xdrfile_xtc.h"
}

class HoppingAnalyzerV3 {
public:
		struct Config {
				std::string gro_file;
				std::string xtc_file;
				std::string output_file = "results.csv";
				float threshold = 0.05f;
				int tw_frames = 20;
		};

		HoppingAnalyzerV3(Config c) : cfg(c) {}

		std::vector<int> getOxygenIndices() {
				std::vector<int> idx;
				std::ifstream f(cfg.gro_file);
				if (!f) return idx;
				std::string line;
				std::getline(f, line); std::getline(f, line);
				int total = std::stoi(line);
				for (int i = 0; i < total; ++i) {
						std::getline(f, line);
						if (line.substr(5, 5).find("SOL") != std::string::npos &&
								line.substr(10, 5).find("O") != std::string::npos) {
								idx.push_back(i);
						}
				}
				return idx;
		}

		void execute() {
				auto o_indices = getOxygenIndices();
				if (o_indices.empty()) { std::cerr << "错误: 未找到 SOL O 原子\n"; return; }
				int n_sol = o_indices.size();

				int natoms;
				char* xtc_c = const_cast<char*>(cfg.xtc_file.c_str());
				if (read_xtc_natoms(xtc_c, &natoms) != 0) { std::cerr << "错误: 无法读取 XTC 文件\n"; return; }

				XDRFILE* xd = xdrfile_open(xtc_c, "r");
				std::vector<float> all_coords;
				std::vector<rvec> coords(natoms);
				matrix box; float time, prec; int step;
				int total_frames = 0;

				std::cout << "1. 正在读取轨迹..." << std::endl;
				while(read_xtc(xd, natoms, &step, &time, box, coords.data(), &prec) == 0) {
						for(int idx : o_indices) {
								all_coords.push_back(coords[idx][0]);
								all_coords.push_back(coords[idx][1]);
								all_coords.push_back(coords[idx][2]);
						}
						total_frames++;
				}
				xdrfile_close(xd);
				if (total_frames == 0) { std::cerr << "错误: 轨迹中没有帧\n"; return; }
				std::cout << "载入完成: " << total_frames << " 帧, " << n_sol << " 原子/帧" << std::endl;

				// 2. 环境初始化
				std::vector<cl::Platform> platforms;
				cl::Platform::get(&platforms);
				std::vector<cl::Device> devices;
				platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
				cl::Context context(devices[0]);
				cl::CommandQueue queue(context, devices[0]);

				// 3. Kernel
				std::string src = R"(
						__kernel void calc_hopping_full(__global const float* coords,
																					 __global float* res,
																					 int n_sol, int total_frames, int hw) {
								int i = get_global_id(0);
								int t = get_global_id(1);

								if(i >= n_sol || t < hw || t >= (total_frames - hw)) return;

								float s1 = 0.0f, s2 = 0.0f;
								// 计算左窗口均值
								for(int j=1; j<=hw; j++) {
										int idx = ((t - j) * n_sol + i) * 3;
										float x = coords[idx]; float y = coords[idx+1]; float z = coords[idx+2];
										s1 += sqrt(x*x + y*y + z*z);
								}
								// 计算右窗口均值
								for(int j=1; j<=hw; j++) {
										int idx = ((t + j) * n_sol + i) * 3;
										float x = coords[idx]; float y = coords[idx+1]; float z = coords[idx+2];
										s2 += sqrt(x*x + y*y + z*z);
								}

								float a1 = s1/hw, a2 = s2/hw;
								float ta = 0.0f, tb = 0.0f;

								// 计算交叉方差
								for(int j=1; j<=hw; j++) {
										int idx1 = ((t - j) * n_sol + i) * 3;
										float x1 = coords[idx1]; float y1 = coords[idx1+1]; float z1 = coords[idx1+2];
										float v1 = sqrt(x1*x1 + y1*y1 + z1*z1);

										int idx2 = ((t + j) * n_sol + i) * 3;
										float x2 = coords[idx2]; float y2 = coords[idx2+1]; float z2 = coords[idx2+2];
										float v2 = sqrt(x2*x2 + y2*y2 + z2*z2);

										ta += (v1 - a2)*(v1 - a2);
										tb += (v2 - a1)*(v2 - a1);
								}
								res[t * n_sol + i] = sqrt((ta/hw)*(tb/hw));
						}
				)";

				cl::Program program(context, src);
				if (program.build({devices[0]}, "-cl-std=CL3.0") != CL_SUCCESS) {
						std::cerr << "编译错误: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
						return;
				}
				cl::Kernel kernel(program, "calc_hopping_full");

				// 4. 显存分配
				cl::Buffer buf_coords(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, all_coords.size() * sizeof(float), all_coords.data());
				cl::Buffer buf_res(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(float));

				// 5. 执行
				int hw = cfg.tw_frames / 2;
				kernel.setArg(0, buf_coords);
				kernel.setArg(1, buf_res);
				kernel.setArg(2, n_sol);
				kernel.setArg(3, total_frames);
				kernel.setArg(4, hw);

				std::cout << "2. GPU 正在运行计算..." << std::endl;
				cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_sol, total_frames));
				if (err != CL_SUCCESS) { std::cerr << "运行错误代码: " << err << std::endl; return; }

				std::vector<float> results(n_sol * total_frames, 0.0f);
				queue.enqueueReadBuffer(buf_res, CL_TRUE, 0, results.size() * sizeof(float), results.data());

				// 6. 结果输出检查
				std::cout << "3. 正在保存结果..." << std::endl;
				std::ofstream csv(cfg.output_file);
				csv << "atom_id,frame,hopping_value\n";
				int count = 0;
					for(int i = 0; i < n_sol; ++i) {
							for(int t = hw; t < total_frames - hw; ++t) {
								float val = results[t * n_sol + i];
								if(val > cfg.threshold) {
										csv << o_indices[i] << "," << t << "," << std::fixed << std::setprecision(5) << val << "\n";
										count++;
								}
						}
				}
				std::cout << "完成！共检测到 " << count << " 次跳跃行为。" << std::endl;
		}

private:
		Config cfg;
};

int main(int argc, char** argv) {
		if (argc < 3) {
				std::cout << "用法: " << argv[0] << " <system.gro> <traj.xtc> [threshold] [output.csv]\n";
				return 1;
		}
		HoppingAnalyzerV3::Config cfg;
		cfg.gro_file = argv[1];
		cfg.xtc_file = argv[2];
		if (argc >= 4) cfg.threshold = std::stof(argv[3]);
		if (argc >= 5) cfg.output_file = argv[4];

		HoppingAnalyzerV3 analyzer(cfg);
		analyzer.execute();
		return 0;
}
