
/*================================================================================
 * Filename: simdCorrelatorMex.cpp
 * Description: This function implementes the correlation function for tracking loops.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Jan, 20, 2020
 *
 * Input:
 *   -- short  *rawSignal: Input IF signal
 *   -- short  *caCode: GPS L1 C/A code sequence
 *   -- double samplingPhase: carrier sampling interval (cycle: Hz*t)
 *   -- double remCarrPhase: initial phase (rad)
 *   -- double remCodePhase: initial code phase(chip)
 *   -- double codePhaseStep: code sampling interval(chip)
 *   -- double codePhaseOffset: half of the early - late code correlation spacing(chips)
 * Output:
 *   -- double *correValues: outputs of correlator values (IE IP IL QE QP QL)
 *==============================================================================*/

#include "pch.h"
#include<stddef.h>
#include<stdio.h>
#include <iostream>
#include<stdlib.h>
#include <chrono>
/* ----- SIMD (SSE2) ----- */
#define SSE2
#if defined(SSE2)
#include <emmintrin.h> 
#include <tmmintrin.h>
#endif
 /* ----- SIMD (AVX) ----- */
#define AVX
#if defined(AVX)
#include <immintrin.h>
#endif

/* Constants ---------------------------------------------------------------------*/
#define PI          3.1415926535897932  /* pi */
#define DPI         (2.0*PI)            /* 2*pi */
#define CDIV        32                  /* carrier lookup table divided cycle */
#define CMASK       0x1F                /* carrier lookup table mask */
#define CSCALE      (1.0/16.0)          /* carrier lookup table scale (LSB) */
               
/* ------------- To preserve variables between mex function calls  ------------- */
static short* iBasebandSignal, * qBasebandSignal; 
static short** localCode;    
static double* correValues;     

#define DEBUG
#if defined(DEBUG)

#endif

/* --------------------- Allocate memory --------------------- */
void allocMemory(int blksize)
{
	correValues = (double*)malloc(6 * sizeof(double));

	/* I & Q signals with carrier wiped off */
	iBasebandSignal = (short*)malloc(sizeof(short) * (blksize + 10));  // 10 is enough for different correlating 
	qBasebandSignal = (short*)malloc(sizeof(short) * (blksize + 10));
	
	/* For local samping code (Early, Prompt and Late) */
	localCode = (short**)malloc(sizeof(short*) * 3);
	for (int i = 0; i < 3; i++)
	{
		localCode[i] = (short*)malloc(sizeof(short) * (static_cast<unsigned long long>(blksize) + 10));
	}
	return;
}

/* -------- Carrier wiping off, code generation and correlation -------- */
double* cFunc_cpuCorrelatorReal(short* rawSignal, short* caCode, double* array, int blksize,int codeLen)
{		
	double remCodePhase = array[0];
	double codePhaseStep = array[1];
	double codePhaseOffset = array[2];
	double remCarrPhase = array[3];
	double samplingPhase = array[4];
	/* -------- Carrier wiping off------- */
	mixCarr(rawSignal, samplingPhase, remCarrPhase, blksize, iBasebandSignal, qBasebandSignal);

	/* -------- code generation------- */
	localCodeGen(caCode, codeLen, remCodePhase, codePhaseStep, codePhaseOffset, blksize, localCode);

	/* -------- correlation----------- */
	correlator(iBasebandSignal, qBasebandSignal, localCode, blksize, correValues);

	return correValues;
}


void cleanup(void)
{
	printf("   file is terminating, destroying allocated memory ...\n");
	/* --------------- Free memory --------------- */
	for (int i = 0; i < 3; i++) {
		free(localCode[i]);
	}
	free(localCode);
	free(iBasebandSignal);
	free(qBasebandSignal);
	free(correValues);
}

/* Generate local code ----------------------------------------------------------
* Generate local code sequences
* Args   : short   *caCode          I   GPS L1 C/A code sequence
*          mwSize codeLen           I   chip number of the C/A code
*          double remCodePhase      I   initial code phase (chip)
*          double codePhaseStep     I   code sampling interval (chip)
*          double codePhaseOffset   I   half of the early-late code correlation spacing (chips)
*          int    blksize           I   number of samples to be generated
*          short  *localCode        O   local code replica outputs
* return : None
*------------------------------------------------------------------------------*/
void localCodeGen(const short* caCode, double codeLen, double remCodePhase, double codePhaseStep,
	double codePhaseOffset, int blksize, short** localCode)
{
#if defined(GENERIC)
	/* code phase of the first sampling point: 1.0 is due to the first code chip added to
	caCode for early replica generation */
	short* p;
	remCodePhase += codePhaseOffset + 1.0;
	if (remCodePhase >= codeLen)
		remCodePhase -= floor(remCodePhase / codeLen) * codeLen;	
	for (p = localCode; p < localCode + blksize; p++, remCodePhase += codePhaseStep)
	{
		if (remCodePhase >= codeLen)
			remCodePhase -= codeLen;
		//*p = caCode[(int)remCodePhase];
		*p = (short)remCodePhase;
	}
	
#elif defined(AVX)
	/* Initial code phase of early code: 1.0 is due to the first code
	chip added to caCode for early replica generation */
	remCodePhase = remCodePhase - codePhaseOffset + 1.0;
	if (remCodePhase >= codeLen)
		remCodePhase -= floor(remCodePhase / codeLen) * codeLen;
	
	short* pE = localCode[0], * pP = localCode[1], * pL = localCode[2];
	int index[16] = { 0 };

	__m256i Index_reg1, Index_reg2;
	__m256 codePhase_reg1, codePhase_reg2;
	__m256 stepNumber_reg = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
	const __m256 remCodePhase_reg = _mm256_set1_ps((float)remCodePhase);
	const __m256 codePhaseStep_reg = _mm256_set1_ps((float)codePhaseStep);
	const __m256 codePhaseOffset_reg = _mm256_set1_ps((float)codePhaseOffset);
	const __m256 sixteenStep = _mm256_set1_ps(16.0f);
	const __m256 eightCodePhase = _mm256_set1_ps((float)(codePhaseStep * 8.0));
	

	for (; pE <= localCode[0] + blksize - 16; pE += 16, pP += 16, pL += 16)
	{
		/* ----------------------- Early code generation ----------------------- */
		codePhase_reg1 = _mm256_fmadd_ps(codePhaseStep_reg, stepNumber_reg, remCodePhase_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg1, eightCodePhase);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		//Store the results
		_mm256_store_si256((__m256i*)index, Index_reg1);
		_mm256_store_si256((__m256i*)(index + 8), Index_reg2);
		for (int i = 0; i < 16; i++)
			pE[i] = caCode[index[i]];

		/* ----------------------- Prompt code generation ----------------------- */
		codePhase_reg1 = _mm256_add_ps(codePhase_reg1, codePhaseOffset_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg2, codePhaseOffset_reg);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		//Store the results
		_mm256_store_si256((__m256i*)index, Index_reg1);
		_mm256_store_si256((__m256i*)(index + 8), Index_reg2);
		for (int i = 0; i < 16; i++)
			pP[i] = caCode[index[i]];

		/* ----------------------- Late code generation ----------------------- */
		codePhase_reg1 = _mm256_add_ps(codePhase_reg1, codePhaseOffset_reg);
		codePhase_reg2 = _mm256_add_ps(codePhase_reg2, codePhaseOffset_reg);
		Index_reg1 = _mm256_cvttps_epi32(codePhase_reg1);
		Index_reg2 = _mm256_cvttps_epi32(codePhase_reg2);
		//Store the results
		_mm256_store_si256((__m256i*)index, Index_reg1);
		_mm256_store_si256((__m256i*)(index + 8), Index_reg2);
		for (int i = 0; i < 16; i++)
			pL[i] = caCode[index[i]];

		/* ------------------ Update the step number register ------------------- */
		stepNumber_reg = _mm256_add_ps(stepNumber_reg, sixteenStep);
	}

	/* ------------ For sampling pionts outside the SIMD interation  ------------ */
	double corrCodePhase = ((double)blksize / 16) * 16 * codePhaseStep + remCodePhase;
	double twoCodePhaseOffset = codePhaseOffset * 2;
	for (; pE < localCode[0] + blksize; pE++, pP++, pL++)
	{
		pE[0] = caCode[(int)corrCodePhase];
		pP[0] = caCode[(int)(corrCodePhase + codePhaseOffset)];
		pL[0] = caCode[(int)(corrCodePhase + twoCodePhaseOffset)];
		corrCodePhase = corrCodePhase + codePhaseStep;
	}

}

/* Mix local carrier ------------------------------------------------------------
* Mix local carrier to input signal
* Args   : short   *rawSignal         I   Input IF signal
*          double samplingPhase       I   carrier sampling interval (cycle: rad*t)
*          double remCarrPhase        I   initial carrier phase (rad)
*          mwSize blksize             I   number of samples to be generated
*          short  *iBasebandSignal    O   I component of input signal with carrier wiped off
*          short  *qBasebandSignal    O   Q component of input signal with carrier wiped off
* Return : None
*------------------------------------------------------------------------------*/
void mixCarr(const short* rawSignal, double samplingPhase, double remCarrPhase, int blksize,
	short* iBasebandSignal, short* qBasebandSignal)
{	
#if defined(GENERIC)
	static short cost[CDIV] = { 0 };            /* carrier lookup table cos(t) */
	
	static short sint[CDIV] = { 0 };           /* carrier lookup table sin(t) */
	const short* IFData;
	double carrPhase, phaseStep;
	mwSize index;

	/* initialize local carrier table */
	if (!cost[0])
	{
		for (int i = 0; i < CDIV; i++)
		{
			cost[i] = (short)floor((cos(DPI / CDIV * i) / CSCALE + 0.5));
			sint[i] = (short)floor((sin(DPI / CDIV * i) / CSCALE + 0.5));
		}
	}


	carrPhase = remCarrPhase * CDIV / DPI;
	phaseStep = samplingPhase * CDIV; /* phase step */
	
	for (IFData = rawSignal; IFData < rawSignal + blksize; IFData++, iBasebandSignal++, qBasebandSignal++, carrPhase += phaseStep)
	{
		index = ((int)carrPhase) & CMASK;
	
		*iBasebandSignal = cost[index] * IFData[0];
		*qBasebandSignal = sint[index] * IFData[0];
	}

#elif defined(AVX)
	const short* IFData;
	double phaseStep;
	/* Inphase and quadrature carrier lookup table */
	static char cost[16] = { 0 }, sint[16] = { 0 };

	/* Carrier lookup table */
	if (!cost[0])
	{
		for (int i = 0; i < 16; i++)  
		{
			cost[i] = (char)floor((cos(PI / 8 * i) / CSCALE + 0.5));
			sint[i] = (char)floor((sin(PI / 8 * i) / CSCALE + 0.5));
		}
	}
	
	remCarrPhase = remCarrPhase * 16 / DPI;
	/* Carrier phase step */
	phaseStep = samplingPhase / DPI * 16;      /* phase step */

	//#pragma pack (32)  

	__m256i Index_mm256, Index_reg1, Index_reg2, cos_reg_mm256, sin_reg_mm256, IF_reg, iBaseband_reg, qBaseband_reg;
	__m256 carrPhase_reg1, carrPhase_reg2;
	__m128i Index_reg, cos_reg_mm128, sin_reg_mm128;	
	/* take into account the packing order of _mm256_packs_epi32 */
	__m256 stepNumber_reg = _mm256_set_ps(11.0f, 10.0f, 9.0f, 8.0f, 3.0f, 2.0f, 1.0f, 0.0f);
	const __m256 remCarrPhase_reg = _mm256_set1_ps((float)remCarrPhase);
	const __m256 carrPhaseStep_reg = _mm256_set1_ps((float)phaseStep);
	const __m256i mask4 = _mm256_set1_epi32(15);
	const __m256 sixteenStep = _mm256_set1_ps(16.0f);
	const __m256 fourPhase = _mm256_set1_ps((float)(phaseStep * 4.0));
	const __m128i local_cos = _mm_loadu_si128((__m128i*)cost);
	const __m128i local_sin = _mm_loadu_si128((__m128i*)sint);
	const __m256i mask_16_to_8 = _mm256_set_epi8(0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
		0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00, \
		0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
		0x0E, 0x0C, 0x0A, 0x08, 0x06, 0x04, 0x02, 0x00);
	
	for (IFData = rawSignal; IFData <= rawSignal + blksize - 16; IFData += 16, iBasebandSignal += 16, qBasebandSignal += 16)
	{
		
		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Carrier phases of the first and second resigters */
		carrPhase_reg1 = _mm256_fmadd_ps(carrPhaseStep_reg, stepNumber_reg, remCarrPhase_reg);
		carrPhase_reg2 = _mm256_add_ps(carrPhase_reg1, fourPhase);
		
		/* Lookup table indexes for the two registers */
		Index_reg1 = _mm256_cvtps_epi32(carrPhase_reg1);   // using _mm256_cvttps_epi32 will result in larger error    
		Index_reg1 = _mm256_and_si256(Index_reg1, mask4);
		Index_reg2 = _mm256_cvtps_epi32(carrPhase_reg2);
		Index_reg2 = _mm256_and_si256(Index_reg2, mask4);
	
		/* Combine the two 32-bit indexes to a single 16-bit index */
		Index_mm256 = _mm256_packs_epi32(Index_reg1, Index_reg2);
		
		/* Convert the 16-bit index of _m256i type to 8-bit index of _m128i type */
		Index_mm256 = _mm256_shuffle_epi8(Index_mm256, mask_16_to_8);
		Index_mm256 = _mm256_permute4x64_epi64(Index_mm256, 0x58);
		Index_reg = _mm256_castsi256_si128(Index_mm256);
		
		/* Form inphase and quadrature carriers from lookup tables */
		cos_reg_mm128 = _mm_shuffle_epi8(local_cos, Index_reg);
		sin_reg_mm128 = _mm_shuffle_epi8(local_sin, Index_reg);
		
		/* Covert the 8-bit carriers to 16-bit type */
		cos_reg_mm256 = _mm256_cvtepi8_epi16(cos_reg_mm128);
		sin_reg_mm256 = _mm256_cvtepi8_epi16(sin_reg_mm128);
		
		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		/* Load input IF signals into SIMD register */
		IF_reg = _mm256_loadu_si256((__m256i*)(IFData));
		
		/* Wiping off carrier from the input IF signals */
		iBaseband_reg = _mm256_mullo_epi16(IF_reg, sin_reg_mm256);
		qBaseband_reg = _mm256_mullo_epi16(IF_reg, cos_reg_mm256);
		
		_mm256_store_si256((__m256i*)(iBasebandSignal), iBaseband_reg);
		_mm256_store_si256((__m256i*)(qBasebandSignal), qBaseband_reg);
		
		stepNumber_reg = _mm256_add_ps(stepNumber_reg, sixteenStep);
		
	}

	/* ----------------- For sampling pionts outside the SIMD interation  ----------------- */
	remCarrPhase = ((double)blksize / 16) * 16 * phaseStep + remCarrPhase;
	int n;
	for (; IFData < rawSignal + blksize; IFData++, iBasebandSignal++, qBasebandSignal++)
	{
		n = ((int)remCarrPhase) % 16;
		iBasebandSignal[0] = sint[n] * IFData[0];
		qBasebandSignal[0] = cost[n] * IFData[0];
		remCarrPhase = remCarrPhase + phaseStep;
	}
#endif
}

/* Correlating function ------------------------------------------------------------
 * Mix local carrier to input signal
 * Args :   short  *iBasebandSignal    I   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    I   Q component of input signal with carrier wiped off
 *          short  **localCode         I   local code sequences
 *          mwSize blksize             I   number of samples to be generated
 *          double *correValues        O   outputs of correlator values
 * Return : None
 *------------------------------------------------------------------------------*/
void correlator(short* iBasebandSignal, short* qBasebandSignal, short** localCode, int blksize, double* correValues)
{

	int I_E_sum[8], I_P_sum[8], I_L_sum[8], Q_E_sum[8], Q_P_sum[8], Q_L_sum[8];
	int I_E = 0, I_P = 0, I_L = 0, Q_E = 0, Q_P = 0, Q_L = 0;
	short* pI, * pQ, * pE, * pP, * pL;
	pI = iBasebandSignal;
	pQ = qBasebandSignal;
	pE = localCode[0];
	pP = localCode[1];
	pL = localCode[2];

	__m256i I_reg, Q_reg, I_E_reg, I_P_reg, I_L_reg, Q_E_reg, Q_P_reg, Q_L_reg, EPL_reg, mul_reg;
	I_E_reg = _mm256_setzero_si256();
	I_P_reg = _mm256_setzero_si256();
	I_L_reg = _mm256_setzero_si256();
	Q_E_reg = _mm256_setzero_si256();
	Q_P_reg = _mm256_setzero_si256();
	Q_L_reg = _mm256_setzero_si256();
	
	
	
	for (int i = 0; pE <= localCode[0] + blksize - 16; pE += 16, pP += 16, pL += 16, pI += 16, pQ += 16, i++) {
		
		/* --------------- Load IF signal with carrier wiped off --------------- */
		I_reg = _mm256_loadu_si256((__m256i*)(pI));
		Q_reg = _mm256_loadu_si256((__m256i*)(pQ));
		
		/* --------------- Compute the early correlation value for data channel --------------- */
		EPL_reg = _mm256_loadu_si256((__m256i*)(pE));
		/* ---- I_E ---- */
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);  //Multiplies signed packed 16-bit integer data elements of two vectors. 
		I_E_reg = _mm256_add_epi32(I_E_reg, mul_reg);
		/* ---- Q_E ---- */
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_E_reg = _mm256_add_epi32(Q_E_reg, mul_reg);

		/* --------------- Compute the prompt correlation value for data channel --------------- */
		EPL_reg = _mm256_loadu_si256((__m256i*)(pP));
		/* ---- I_P ---- */
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);
		I_P_reg = _mm256_add_epi32(I_P_reg, mul_reg);
		/* ---- Q_P ---- */
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_P_reg = _mm256_add_epi32(Q_P_reg, mul_reg);

		/* --------------- Compute the late correlation value for data channel --------------- */
		EPL_reg = _mm256_loadu_si256((__m256i*)(pL));
		/* ---- I_L ---- */
		mul_reg = _mm256_madd_epi16(I_reg, EPL_reg);
		I_L_reg = _mm256_add_epi32(I_L_reg, mul_reg);
		/* ---- Q_L ---- */
		mul_reg = _mm256_madd_epi16(Q_reg, EPL_reg);
		Q_L_reg = _mm256_add_epi32(Q_L_reg, mul_reg);
	}
	
	/* --------------- Integation for SIMD register elements --------------- */
	_mm256_store_si256((__m256i*)I_E_sum, I_E_reg);
	_mm256_store_si256((__m256i*)Q_E_sum, Q_E_reg);
	_mm256_store_si256((__m256i*)I_P_sum, I_P_reg);
	_mm256_store_si256((__m256i*)Q_P_sum, Q_P_reg);
	_mm256_store_si256((__m256i*)I_L_sum, I_L_reg);
	_mm256_store_si256((__m256i*)Q_L_sum, Q_L_reg);
	for (int i = 0; i < 8; i++) {
		I_E += I_E_sum[i]; Q_E += Q_E_sum[i];
		I_P += I_P_sum[i]; Q_P += Q_P_sum[i];
		I_L += I_L_sum[i]; Q_L += Q_L_sum[i];
	}
	
	/* ----------------- For sampling pionts outside the SIMD interation  ----------------- */
	
	for (int i = 0; pE < localCode[0] + blksize; pI++, pQ++, pE++, pP++, pL++, i++) {
		I_E += pI[0] * pE[0]; Q_E += pQ[0] * pE[0];
		I_P += pI[0] * pP[0]; Q_P += pQ[0] * pP[0];
		I_L += pI[0] * pL[0]; Q_L += pQ[0] * pL[0];
	}

	/* Convert to double data type */
	correValues[0] = I_E * CSCALE; correValues[1] = Q_E * CSCALE;
	correValues[2] = I_P * CSCALE; correValues[3] = Q_P * CSCALE;
	correValues[4] = I_L * CSCALE; correValues[5] = Q_L * CSCALE;

}


#endif