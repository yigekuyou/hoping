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
#include <QDebug>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QVector>
#include <QStringView>
#include <QFile>
#include <QTextStream>
#include <QtConcurrent>
#include <QFuture>
#include <QCoreApplication>
#include <QStringList>
#include <QFileInfo>
extern "C" {
		#include "xdrfile/xdrfile_xtc.h"
}

#include <algorithm>
#include <set>

class HoppingAnalyzer {
public:
		struct Config {
				QString gro_file;
				QString xtc_file;
				QString output_file = "results.csv";
				int tw_frames = 20;
		};

		HoppingAnalyzer(Config c) : cfg(c) {}

		QVector<int> getOxygenIndices() {
				//使用 QFile 打开文件
				QFile file(cfg.gro_file);
				if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
						qCritical() << "错误: 无法打开文件" << file.fileName();
						return {};
				}
				QTextStream in(&file);
				//读取前两行
				QString line = in.readLine(); // 标题行
				if (line.isNull()) return {};
				line = in.readLine(); // 原子总数行
				if (line.isNull()) return {};
				bool ok;
				int total = line.trimmed().toInt(&ok);
				if (!ok) return {};
				QVector<int> idx;
				idx.reserve(total);
				//循环读取原子信息
				for (int i = 0; i < total; ++i) {
						line = in.readLine();
						if (line.isNull() || line.length() < 20) break;
						// GRO 格式固定列宽：
						// 残基序号(5位), 残基名(5位), 原子名(5位), 原子序号(5位)...
						QString residueName = line.mid(5, 5).trimmed();
						QString atomName = line.mid(10, 5).trimmed();
						// 调试输出：检查实际裁剪后的字符串
						 qDebug() << "Index:" << i << "Res:" << residueName << "Atom:" << atomName;
						// 匹配
						if (residueName == "R03" && atomName.startsWith('O')) {
								idx.push_back(i);
						}
				}
				return idx;
		}
		void execute() {
			QVector<int> o_indices = getOxygenIndices();
			if (o_indices.isEmpty()) {
					qCritical() << "错误: 未找到符合条件的原子";
					return;
			}
				int n_sol = o_indices.size();
				int hw = cfg.tw_frames / 2;
				int natoms= 0;

				QByteArray xtcPath = cfg.xtc_file.toLocal8Bit();
				char* xtc_c = xtcPath.data();
				XDRFILE* xd = xdrfile_open(xtc_c, "r");
				if (!xd) {
						qCritical() << "错误: 无法打开 XTC 文件";
						return;
				}
				if (read_xtc_natoms(xtc_c, &natoms) != 0) {
								qCritical() << "错误: 无法读取 XTC 原子总数";
								return;
				}
				std::vector<float> all_coords;
				std::vector<rvec> coords(natoms);
				matrix box; float time, prec; int step;
				int total_frames = 0;

				qInfo() << "1. 正在读取轨迹...";;
				while(read_xtc(xd, natoms, &step, &time, box, coords.data(), &prec) == 0) {
						qDebug()<< "Processing Frame:" << total_frames
										<< "First Atom:" << coords[0][0]
										<< "," << coords[0][1]
										<<"," << coords[0][2]
										<< "Time:" << time << "ps";
						for(int idx : o_indices) {
								all_coords.push_back(coords[idx][0]*10);
								all_coords.push_back(coords[idx][1]*10);
								all_coords.push_back(coords[idx][2]*10);
						}
						total_frames++;
				}
				xdrfile_close(xd);
				if (total_frames == 0) {
						qCritical() << "错误: 轨迹文件中没有读入任何帧";
						return;
				}
				qInfo().nospace() << "载入完成: " << total_frames << " 帧, "
													<< n_sol << " 个计算原子/帧 (总坐标数: " << all_coords.size() << ")";
				// 2. 环境初始化
				cl::Context context(CL_DEVICE_TYPE_GPU);
				cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0]);

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

							hopping_val = native_sqrt(MS_A_to_B  * MS_B_to_A );
							}
						res[t * n_sol + i] = (float2)(hopping_val, dist);
						}

				)";
				const std::string optics_time_src = R"(
				)";
				cl::Program prog_hop(context, hopping_src);
								prog_hop.build("-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math");
								cl::Kernel k_hop(prog_hop, "calc_hopping_full");

/*
				cl::Program prog_optics(context, optics_time_src);
								prog_optics.build(std::format("-D MIN_PTS={} -cl-std=CL3.0 -cl-fast-relaxed-math", cfg.tw_frames));
								cl::Kernel k_optics(prog_optics, "calc_time_core_dist");
*/
				// 4. 显存分配
				cl::Buffer buf_coords(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, all_coords.size() * sizeof(float), all_coords.data());
				cl::Buffer buf_res(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(cl_float2));
//				cl::Buffer buf_core(context, CL_MEM_WRITE_ONLY |CL_MEM_HOST_READ_ONLY, n_sol * total_frames * sizeof(float));
				// 5. 执行
				k_hop.setArg(0, buf_coords);
				k_hop.setArg(1, buf_res);
				k_hop.setArg(2, n_sol);
				k_hop.setArg(3, total_frames);
				k_hop.setArg(4, hw);
/*
				k_optics.setArg(0, buf_coords);
				k_optics.setArg(1, buf_core);
				k_optics.setArg(2, n_sol);
				k_optics.setArg(3, total_frames);
*/
				qDebug() << "2. GPU 正在运行计算...";
				cl_int hop_err = queue.enqueueNDRangeKernel(k_hop, cl::NullRange, cl::NDRange(n_sol, total_frames));
				if (hop_err != CL_SUCCESS) { std::cerr << "k_hop运行错误代码: " << hop_err << std::endl; return; }
//				cl_int optics_err = queue.enqueueNDRangeKernel(k_optics, cl::NullRange, cl::NDRange(n_sol, total_frames));
//				if (optics_err != CL_SUCCESS) { std::cerr << "k_optics运行错误代码: " << optics_err << std::endl; return; }
						QVector<cl_float2> results(n_sol * total_frames);
						queue.enqueueReadBuffer(buf_res, CL_TRUE, 0, results.size() * sizeof(cl_float2), results.data());
						qDebug() << "3. 正在保存结果...";
						// 准备输出文件
						QFile file(cfg.output_file);
						if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
								qCritical() << "无法打开文件进行写入";
								return;
						}
						QTextStream out(&file);
						out << "原子ID,帧号,跳跃值(Hopping),位移(nm)\n";
						QVector<int> indices(n_sol);
						std::iota(indices.begin(), indices.end(), 0);

						// 并行映射处理：将每个原子的所有帧转换为字符串
						auto mapFunction = [&](int i) -> QString {
								QString block;
								block.reserve(total_frames * 50); // 预分配内存减少开销
								int atom_actual_idx = o_indices[i];

								for (int t = 0; t < total_frames; ++t) {
										cl_float2 res_pair = results[t * n_sol + i];
										// 格式化
										block.append(QString("%1,%2,%3,%4\n")
																 .arg(atom_actual_idx)
																 .arg(t)
																 .arg(res_pair.s[0], 0, 'f', 5)
																 .arg(res_pair.s[1], 0, 'f', 5));
								}
								return block;
						};

						// 执行并行计算并阻塞等待结果
						QList<QString> resultBlocks = QtConcurrent::blockingMapped<QList<QString>>(indices, mapFunction);

						// 将结果写入文件
						for (const QString& block : resultBlocks) {
								out << block;
						}

						file.close();
						qDebug() << "保存完成！";
				}
private:
		Config cfg;
};

int main(int argc, char** argv) {
		//初始化 QCoreApplication，它会自动处理命令行参数的编码转换
		QCoreApplication a(argc, argv);

		//获取参数列表（包含程序名）
		QStringList args = QCoreApplication::arguments();

		if (args.size() < 3) {
				// 使用 qInfo
				std::cout << "用法: " << args.at(0).toLocal8Bit().constData()
									<< " <system.gro> <traj.xtc> [output.csv]" << std::endl;
				return 1;
		}

		//填充配置
		HoppingAnalyzer::Config cfg;

		// Qt 的 arguments 已经处理好了编码，直接赋值给 QString
		cfg.gro_file = args.at(1);
		cfg.xtc_file = args.at(2);

		if (args.size() >= 4) {
				cfg.output_file = args.at(3);
		}

		// 4. 逻辑执行
		HoppingAnalyzer analyzer(cfg);
		analyzer.execute();
		return 0;
}
