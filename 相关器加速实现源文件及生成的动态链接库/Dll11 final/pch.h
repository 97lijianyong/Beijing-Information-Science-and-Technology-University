// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"
#include<stdlib.h>
#include <cstdint>
// 添加要在此处预编译的标头


extern "C" _declspec(dllexport) double* cFunc_cpuCorrelatorReal(short* rawSignal, short* caCode, double* array, int blksize, int codeLen);

//
extern "C" _declspec(dllexport)  void localCodeGen(const short* caCode, double codeLen, double remCodePhase, double codePhaseStep,
	double codePhaseOffset, int blksize, short** localCode);
extern "C" _declspec(dllexport) void cleanup(void);
extern "C" _declspec(dllexport)  void mixCarr(const short* rawSignal, double samplingPhase, double remCarrPhase, int blksize,
	short* iBasebandSignal, short* qBasebandSignal);
extern "C" _declspec(dllexport) void correlator(short* iBasebandSignal, short* qBasebandSignal, short** localCode, int blksize, double* correValues);
extern "C" _declspec(dllexport) void allocMemory(int blksize);
#endif //PCH_H
