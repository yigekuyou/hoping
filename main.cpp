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
				int tw_frames = 100;
		};

		HoppingAnalyzerV3(Config c) : cfg(c) {}

		// 解析 GRO 文件获取 SOL 的 O 原子
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
				if (o_indices.empty()) { std::cerr << "未找到 SOL O 原子\n"; return; }
				int n_sol = o_indices.size();

				// 1. 初始化 OpenCL 3.0 环境
				std::vector<cl::Platform> platforms;
				cl::Platform::get(&platforms);
				if (platforms.empty()) {
						std::cerr << "未找到 OpenCL 平台" << std::endl;
						return;
				}

				std::vector<cl::Device> devices;
				// 这里必须传入 vector 的地址
				platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

				if (devices.empty()) {
						std::cerr << "未找到 GPU 设备" << std::endl;
						return;
				}

				cl::Device device = devices[0]; // 选取第一个 GPU
				cl::Context context(device);
				cl::CommandQueue queue(context, device);
				// 2. 编译 Kernel
				std::string src = R"(
						__kernel void calc_hopping(__global const float* data, __global float* res,
																			int n, int hw, int t_idx, int sz) {
								int i = get_global_id(0);
								if(i >= n) return;
								float s1 = 0, s2 = 0;
								for(int j=1; j<=hw; j++) {
										s1 += data[i + ((t_idx - j + sz) % sz) * n];
										s2 += data[i + ((t_idx + j) % sz) * n];
								}
								float a1 = s1/hw, a2 = s2/hw;
								float ta = 0, tb = 0;
								for(int j=1; j<=hw; j++) {
										float v1 = data[i + ((t_idx - j + sz) % sz) * n];
										float v2 = data[i + ((t_idx + j) % sz) * n];
										ta += (v1 - a2)*(v1 - a2);
										tb += (v2 - a1)*(v2 - a1);
								}
								res[i] = sqrt((ta/hw)*(tb/hw));
						}
				)";
				cl::Program program(context, src);
				program.build("-cl-std=CL3.0");
				cl::Kernel kernel(program, "calc_hopping");

				// 3. 准备显存
				cl::Buffer buf_data(context, CL_MEM_READ_WRITE, sizeof(float) * n_sol * cfg.tw_frames);
				cl::Buffer buf_res(context, CL_MEM_WRITE_ONLY, sizeof(float) * n_sol);

				// 4. 读取轨迹
				int natoms;
				char* xtc_c = const_cast<char*>(cfg.xtc_file.c_str());
				read_xtc_natoms(xtc_c, &natoms);
				XDRFILE* xd = xdrfile_open(xtc_c, "r");
				std::vector<rvec> coords(natoms);
				matrix box; float time, prec; int step;

				std::ofstream csv(cfg.output_file);
				csv << "atom_id,frame,hopping_value\n";

				int frame_count = 0;
				int hw = cfg.tw_frames / 2;
				std::cout << "开始分析轨迹: " << cfg.xtc_file << " (OpenCL 3.0)\n";

				while(read_xtc(xd, natoms, &step, &time, box, coords.data(), &prec) == 0) {
						std::vector<float> mags(n_sol);
						for(int i=0; i<n_sol; ++i) {
								rvec& r = coords[o_indices[i]];
								mags[i] = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
						}

						int pos = frame_count % cfg.tw_frames;
						queue.enqueueWriteBuffer(buf_data, CL_TRUE, pos * n_sol * sizeof(float), n_sol * sizeof(float), mags.data());

						if(frame_count >= cfg.tw_frames) {
								int t_idx = (pos - hw + cfg.tw_frames) % cfg.tw_frames;
								kernel.setArg(0, buf_data); kernel.setArg(1, buf_res);
								kernel.setArg(2, n_sol); kernel.setArg(3, hw);
								kernel.setArg(4, t_idx); kernel.setArg(5, cfg.tw_frames);

								queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_sol));
								std::vector<float> out(n_sol);
								queue.enqueueReadBuffer(buf_res, CL_TRUE, 0, n_sol * sizeof(float), out.data());

								int target_f = frame_count - hw;
								for(int i=0; i<n_sol; ++i) {
										if(out[i] > cfg.threshold) {
												csv << o_indices[i] << "," << target_f << "," << std::fixed << std::setprecision(5) << out[i] << "\n";
										}
								}
						}
						frame_count++;
						if(frame_count % 1000 == 0) std::cout << "当前帧: " << frame_count << "\r" << std::flush;
				}
				xdrfile_close(xd);
				std::cout << "\n分析完成，结果已存至: " << cfg.output_file << std::endl;
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
