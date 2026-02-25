#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <future>
#include <format>
extern "C" {
		#include "xdrfile/xdrfile_xtc.h"
}

class HoppingAnalyzerV3 {
public:
		struct Config {
				std::string gro_file;
				std::string xtc_file;
				std::string output_file = "results.csv";
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
				cl::Context context(CL_DEVICE_TYPE_GPU);
				cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0]);

				// 3. Kernel
				std::string src = R"(
						__kernel void calc_hopping_full(__global const float* coords,
						__global float2* res,
						int n_sol, int total_frames, int hw) {
						int i = get_global_id(0);
						int t = get_global_id(1);
						if(i >= n_sol || t >= total_frames ) return;

						// --- 计算位移 Displacement: |r(t) - r(0)| ---
						int curr_base = (t * n_sol + i) * 3;
						float3 p_curr = vload3(0, &coords[curr_base]);
						float3 p_init = vload3(0, &coords[i * 3]);
						float dist = distance(p_curr, p_init);
						// --- 计算 Hopping Value ---
						float hopping_val = 0.0f;
						if(t >= hw && t < (total_frames - hw)) {
						float ta = 0.0f, tb = 0.0f, avg1 = 0.0f, avg2 = 0.0f;
						for(int j=1; j<=hw; j++) {
						avg1 += length(vload3(0, &coords[((t - j) * n_sol + i) * 3]));
						avg2 += length(vload3(0, &coords[((t + j) * n_sol + i) * 3]));
						}
						avg1 /= hw; avg2 /= hw;
						for(int j=1; j<=hw; j++) {
						float v1 = length(vload3(0, &coords[((t - j) * n_sol + i) * 3]));
						float v2 = length(vload3(0, &coords[((t + j) * n_sol + i) * 3]));
						ta += (v1 - avg2) * (v1 - avg2);
						tb += (v2 - avg1) * (v2 - avg1);
						}
						hopping_val = native_sqrt((ta/hw) * (tb/hw));
						}
						res[t * n_sol + i] = (float2)(hopping_val, dist);
						}
				)";

				cl::Program program(context, src);
								program.build("-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math");
								cl::Kernel kernel(program, "calc_hopping_full");

				// 4. 显存分配
				cl::Buffer buf_coords(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, all_coords.size() * sizeof(float), all_coords.data());
				cl::Buffer buf_res(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(cl_float2));

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

				std::vector<cl_float2> results(n_sol * total_frames);
				queue.enqueueReadBuffer(buf_res, CL_TRUE, 0, results.size() * sizeof(cl_float2), results.data());

				// 6. 结果输出检查
				std::cout << "3. 正在保存结果..." << std::endl;

				const int num_threads = std::thread::hardware_concurrency();
				const int atoms_per_thread = n_sol / num_threads;
				std::vector<std::string> thread_buffers(num_threads);
				std::vector<std::jthread> workers;

				for (int t_id = 0; t_id < num_threads; ++t_id) {
				int start_atom = t_id * atoms_per_thread;
				int end_atom = (t_id == num_threads - 1) ? n_sol : (t_id + 1) * atoms_per_thread;

				workers.emplace_back([&, t_id, start_atom, end_atom]() {
				// 预估缓冲区大小以减少 realloc (每个 entry 约 40-60 字节)
				std::string local_buf;
				local_buf.reserve((end_atom - start_atom) * total_frames * 50);

				for (int i = start_atom; i < end_atom; ++i) {
				int atom_actual_idx = o_indices[i];
				for (int t = 0; t < total_frames; ++t) {
				cl_float2 res_pair = results[t * n_sol + i];

				// 使用 std::format (C++20) 提供比 stringstream 更快的格式化速度
				// 格式：atom_id,frame,hopping,dist
				std::format_to(std::back_inserter(local_buf),
				"{},{},{:.5f},{:.5f}\n",
				atom_actual_idx, t, res_pair.s[0], res_pair.s[1]);
				}
			}
			thread_buffers[t_id] = std::move(local_buf);
		});
	}
	std::ofstream csv(cfg.output_file, std::ios::binary);
	csv << "atom_id,frame,hopping_value,displacement(nm)\n";
	workers.clear();
	for (const auto& buf : thread_buffers) {
		csv.write(buf.data(), buf.size());
	}
	std::cout << "保存完成！" << std::endl;
}

private:
		Config cfg;
};

int main(int argc, char** argv) {
		if (argc < 3) {
				std::cout << "用法: " << argv[0] << " <system.gro> <traj.xtc> [output.csv]\n";
				return 1;
		}
		HoppingAnalyzerV3::Config cfg;
		cfg.gro_file = argv[1];
		cfg.xtc_file = argv[2];
		if (argc >= 4) cfg.output_file = argv[3];

		HoppingAnalyzerV3 analyzer(cfg);
		analyzer.execute();
		return 0;
}
