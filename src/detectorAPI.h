#ifndef __DETECTOR_H__
#define __DETECTOR_H__

#include "darknet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX_DETECT_NUM
#define MAX_DETECT_NUM 200
#endif/*MAX_DETECT_NUM*/

#define GPU 1
network *g_net;

typedef struct Result
{
	int l;
	int r;
	int t;
	int b;
	float x;
	float y;
	float w;
	float h;
	int idx;
	float prob;
}Result_t;

typedef struct Detector
{
	size_t max_model_num;
	size_t model_num;
	char** m_names;
	int m_gpu_index;
	float m_thresh;
	float m_hier_thresh;
	network *m_net;
	int m_layer_num;
	layer m_last_layer;
	int m_last_width;
	int m_last_height;
	int m_last_num;
	int m_last_total_size;
	//box *m_boxes;
	//float** m_probs;
	int resNum;
    Result_t *res;
}Detector_t;

typedef enum _DETECT_RET_{
	DETECT_SUCCESS,
	DETECT_CUDA_ERR,
	DETECT_INIT_ERR,
	DETECT_IMAGE_ERR,
	DETECT_RESET_ERR,
	DETECT_ERR,
	DETECT_DIST_ERR,
	DETECT_POSTPROCESS_RESULT_ERR
}DETECT_RET;

#ifdef __cplusplus
extern "C" {
#endif
DETECT_RET det_init(const char *cfgFile, const char *weightFile, const char *nameFile, Detector_t *detector);
DETECT_RET det_img_load(const char *imgFile, image *im);
DETECT_RET detect_reset(Detector_t *detector);
DETECT_RET detect(Detector_t *detector, const image im);
//DETECT_RET detect_get_result(Detector *detector, Result_t *res);
DETECT_RET detect_destroy(Detector_t *detector);
#ifdef __cplusplus
}
#endif

#endif /*__DETECTOR_H__*/
