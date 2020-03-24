#include "detectorAPI.h"

DETECT_RET det_init(const char *cfgFile, const char *weightFile, const char *nameFile, Detector_t *detector)
{
	if (cfgFile == NULL || weightFile == NULL || detector == NULL)
	{
		return DETECT_INIT_ERR;
	}
	cuda_set_device(0);
    //g_net = parse_network_cfg(cfgFile);
    detector->m_net = parse_network_cfg(cfgFile);
    if(weightFile && weightFile[0] != 0){
        load_weights(detector->m_net, weightFile);
    }
    set_batch_network(detector->m_net, 1);
	detector->m_names = get_labels(nameFile);
	detector->m_thresh = 0.1;
	detector->m_hier_thresh = 0.5;
	detector->res = NULL;
    detector->resNum = 0;
	return DETECT_SUCCESS;
}


DETECT_RET det_img_load(const char *imgFile, image *im)
{
	if (imgFile == NULL || im == NULL)
	{
		return DETECT_IMAGE_ERR;
	}

	int ret = load_image_color2(imgFile, 0, 0, im);
    if (ret != 0)
        return DETECT_IMAGE_ERR;
	return DETECT_SUCCESS;
}


DETECT_RET detect_reset(Detector_t *detector)
{
	if (detector == NULL)
	{
		return DETECT_RESET_ERR;
	}
	if (detector->res != NULL)
	{
		free(detector->res);
		detector->res = NULL;
        detector->resNum = 0;
	}
	return DETECT_SUCCESS;
}


DETECT_RET detect(Detector_t *detector, const image im)//  Result_t **detRes
{
	if (detector == NULL)
	{
		return DETECT_ERR;
	}
	image **alphabet = NULL;
	image sized = letterbox_image(im, detector->m_net->w, detector->m_net->h);
	layer l = detector->m_net->layers[detector->m_net->n-1];
	float *X = sized.data;
	network_predict(detector->m_net, X);

	int nboxes = 0;
	float nms=.45;

	detection *dets = get_network_boxes(detector->m_net, im.w, im.h, 
		detector->m_thresh, detector->m_hier_thresh, 0, 1, &nboxes);
	/* if (nms) do_nms_sort(dets, nboxes, l.classes, nms); */
    if (nms)
        diounms_sort(dets, nboxes, l.classes, nms, "iou", "greedynms");
	Res_t *res = calloc(nboxes, sizeof(Res_t));
	//*detRes = calloc(nboxes, sizeof(Result_t));
	detector->res = calloc(nboxes, sizeof(Result_t));
	int ii = 0;
	for (ii = 0; ii < nboxes; ++ii)
		res[ii].cls = -1;
    float thresh = detector->m_thresh;
	draw_detections_top1(im, dets, nboxes, &thresh, detector->m_names, alphabet, l.classes, res);
    int validedResCount = 0;
	for (ii = 0; ii < nboxes; ++ii)
	{
		if (res[ii].cls == -1)
			continue;
        (detector->res)[validedResCount].idx = res[ii].cls;
		(detector->res)[validedResCount].l = res[ii].l;
		(detector->res)[validedResCount].r = res[ii].r;
		(detector->res)[validedResCount].t = res[ii].t;
		(detector->res)[validedResCount].b = res[ii].b;
		(detector->res)[validedResCount].prob = res[ii].prob;
        (detector->res)[validedResCount].x = res[ii].x;
        (detector->res)[validedResCount].y = res[ii].y;
        (detector->res)[validedResCount].w = res[ii].w;
        (detector->res)[validedResCount].h = res[ii].h;
        validedResCount += 1;
	}
	//detector->res = *detRes;
    detector->resNum = validedResCount;
	if (res != NULL)
	{
		free(res);
		res = NULL;
	}
	free_detections(dets, nboxes);
	free_image(sized);
    free_image(im);

	return DETECT_SUCCESS;
}


DETECT_RET detect_destroy(Detector_t *detector)
{
	if (detector == NULL)
	{
		return DETECT_DIST_ERR;
	}
	if (detector->res != NULL)
	{
		free(detector->res);
		detector->res = NULL;
	}
	return DETECT_SUCCESS;
}
