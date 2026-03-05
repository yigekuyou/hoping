#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <future>
#include <format>
#include <ranges>
#include <string_view>
#include <print>
#include <omp.h>
extern "C" {
		#include "xdrfile/xdrfile_xtc.h"
}

#include <algorithm>
#include <set>

class HoppingAnalyzer {
public:
		struct Config {
				std::string gro_file;
				std::string xtc_file;
				std::string output_file = "results.csv";
				int tw_frames = 20;
		};

		HoppingAnalyzer(Config c) : cfg(c) {}

		std::vector<int> getOxygenIndices() {
				std::ifstream f(cfg.gro_file);
				if (!f) {
						std::println(stderr, "错误: 无法打开文件 {}", cfg.gro_file);
						return {};
				}

				std::string line;
				// 跳过标题行和读取原子总数行
				if (!std::getline(f, line) || !std::getline(f, line)) return {};

				int total = std::stoi(line);
				std::vector<int> idx;
				idx.reserve(total);

				for (int i : std::views::iota(0, total)) {
						if (!std::getline(f, line) || line.length() < 20) break;
						std::string_view sv(line);
						auto residue_name = sv.substr(5, 5);
						auto atom_name = sv.substr(10, 5);

						// 检查是否包含 SOL 和 O (原子)
						if (residue_name.contains("R02") && atom_name.contains('O')) {
								idx.push_back(i);
						}
				}
				return idx;
		}
		void execute() {
				auto o_indices = getOxygenIndices();
				if (o_indices.empty()) { std::cerr << "错误: 未找到 SOL OW 原子\n"; return; }
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
				int hw = cfg.tw_frames / 2;

				// 3. Kernel
				std::string hopping_src = R"(
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
						float3 avg_A = (float3)(0.0f);
						float3 avg_B = (float3)(0.0f);
						if(t >= hw && t < (total_frames - hw)) {
							// 窗口 A [t-hw 到 t] 和 窗口 B [t 到 t+hw] 的平均位置
							for(int j = 0; j <= hw; j++) {
								avg_A += vload3(0, &coords[((t - j) * n_sol + i) * 3]);
								avg_B += vload3(0, &coords[((t + j) * n_sol + i) * 3]);
							}
							avg_A /= (float)(hw + 1) ;
							avg_B /= (float)(hw + 1) ;
							float MS_A_to_B = 0.0f;
							float MS_B_to_A = 0.0f;;

							// 计算hopping
							for(int j = 0; j <= hw; j++) {
								float3 pos_A = vload3(0, &coords[((t - j) * n_sol + i) * 3]);
								float3 pos_B = vload3(0, &coords[((t + j) * n_sol + i) * 3]);

								float3 diffA = pos_A - avg_B;
								MS_A_to_B += dot(diffA, diffA);

								float3 diffB = pos_B - avg_A;
								MS_B_to_A += dot(diffB, diffB);
							}

							hopping_val = native_sqrt((MS_A_to_B / hw) * (MS_B_to_A / hw));
							}
						res[t * n_sol + i] = (float2)(hopping_val, dist);
						}

				)";
				const std::string optics_time_src = R"(
						__kernel void calc_time_core_dist(__global const float* coords,
																							__global float* core_dists,
																							int n_sol,
																							int total_frames
																							){
								int i = get_global_id(0);
								int t = get_global_id(1);
								if(i >= n_sol || t >= total_frames) return;

								float3 p_t = vload3(0, &coords[(t * n_sol + i) * 3]);

								// 查找该原子在整个轨迹中距离 p_t 最近的第 minPts 帧
								float nearest[MIN_PTS];
								for(int k=0; k<MIN_PTS; k++) nearest[k] = 1e10f;

								for(int t_other = 0; t_other < total_frames; t_other++) {
										if(t == t_other) continue;
										float3 p_other = vload3(0, &coords[(t_other * n_sol + i) * 3]);
										float d = distance(p_t, p_other);

										if(d < nearest[MIN_PTS-1]) {
												nearest[MIN_PTS-1] = d;
												for(int m=MIN_PTS-2; m>=0 && nearest[m] > d; m--) {
														nearest[m+1] = nearest[m];
														nearest[m] = d;
												}
										}
								}
								core_dists[t * n_sol + i] = nearest[MIN_PTS-1];
						}
				)";
				cl::Program prog_hop(context, hopping_src);
								prog_hop.build("-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math");
								cl::Kernel k_hop(prog_hop, "calc_hopping_full");
				cl::Program prog_optics(context, optics_time_src);
								prog_optics.build(std::format("-D MIN_PTS={} -cl-std=CL3.0 -cl-fast-relaxed-math", cfg.tw_frames));
								cl::Kernel k_optics(prog_optics, "calc_time_core_dist");

				// 4. 显存分配
				cl::Buffer buf_coords(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, all_coords.size() * sizeof(float), all_coords.data());
				cl::Buffer buf_res(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(cl_float2));
				cl::Buffer buf_core(context, CL_MEM_WRITE_ONLY |CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(float));
				// 5. 执行
				k_hop.setArg(0, buf_coords);
				k_hop.setArg(1, buf_res);
				k_hop.setArg(2, n_sol);
				k_hop.setArg(3, total_frames);
				k_hop.setArg(4, hw);

				k_optics.setArg(0, buf_coords);
				k_optics.setArg(1, buf_core);
				k_optics.setArg(2, n_sol);
				k_optics.setArg(3, total_frames);
				std::cout << "2. GPU 正在运行计算..." << std::endl;
				cl_int hop_err = queue.enqueueNDRangeKernel(k_hop, cl::NullRange, cl::NDRange(n_sol, total_frames));
				if (hop_err != CL_SUCCESS) { std::cerr << "k_hop运行错误代码: " << hop_err << std::endl; return; }
				cl_int optics_err = queue.enqueueNDRangeKernel(k_optics, cl::NullRange, cl::NDRange(n_sol, total_frames));
				if (optics_err != CL_SUCCESS) { std::cerr << "k_optics运行错误代码: " << optics_err << std::endl; return; }

				std::vector<cl_float2> results(n_sol * total_frames);
				std::vector<float> c_data(n_sol * total_frames);
				queue.enqueueReadBuffer(buf_res, CL_TRUE, 0, results.size() * sizeof(cl_float2), results.data());
				queue.enqueueReadBuffer(buf_core, CL_TRUE, 0, c_data.size() * sizeof(float), c_data.data());

				std::vector<float> sorted_dists = c_data;
				std::sort(sorted_dists.begin(), sorted_dists.end());

				// 取第 80 百分位数的数值作为阈值
				float auto_eps = sorted_dists[static_cast<int>(sorted_dists.size() * 0.8)];
				std::println("基于数据分布建议的阈值 eps: {:.5f}", auto_eps);
				std::vector<int> atom_region_counts(n_sol, 0);

				#pragma omp parallel for
				for (int i = 0; i < n_sol; ++i) {
						bool is_in_stable_region = false;
						int count = 0;

						for (int t = 0; t < total_frames; ++t) {
								float core_dist = c_data[t * n_sol + i];

								// OPTICS 核心逻辑：如果核心距离小于设定阈值，认为该点属于一个“簇”
								if (core_dist <= auto_eps) {
										if (!is_in_stable_region) {
												// 发现新区域
												count++;
												is_in_stable_region = true;
										}
								} else {
										// 离开区域
										is_in_stable_region = false;
								}
						}
						atom_region_counts[i] = count;
				}



				std::cout << "3. 正在保存结果..." << std::endl;
				std::ofstream csv(cfg.output_file, std::ios::binary);
				csv << "原子ID,帧号,跳跃值(Hopping),位移(nm),核心距离(Core_Dist),区域总数\n";
				int num_threads = omp_get_max_threads();
				std::vector<std::string> thread_buffers(num_threads);
#pragma omp parallel
{
		int t_id = omp_get_thread_num();
		std::string& local_buf = thread_buffers[t_id];

		local_buf.reserve((n_sol / num_threads) * total_frames * 50);

		#pragma omp for schedule(dynamic)
		for (int i = 0; i < n_sol; ++i) {
				int atom_actual_idx = o_indices[i];
				int total_regions = atom_region_counts[i];
				for (int t = 0; t < total_frames; ++t) {
				cl_float2 res_pair = results[t * n_sol + i];
				float core_d = c_data[t * n_sol + i];
				// 格式：atom_id,frame,hopping,dist
				std::format_to(std::back_inserter(local_buf),
				"{},{},{:.5f},{:.5f},{:.5f},{}\n",
				atom_actual_idx, t, res_pair.s[0], res_pair.s[1],core_d,total_regions);
				}
		}
}

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
		HoppingAnalyzer::Config cfg;
		cfg.gro_file = argv[1];
		cfg.xtc_file = argv[2];
		if (argc >= 4) cfg.output_file = argv[3];

		HoppingAnalyzer analyzer(cfg);
		analyzer.execute();
		return 0;
}
