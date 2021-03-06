#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

/*
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}
*/

ious delta_yolo_box_iouloss(box truth, float *x, float *biases, int n, int index, \
                            int i, int j, int lw, int lh, int w, int h, float *delta, float scale, \
                            int stride, float class_weight, float iou_normalizer, IOU_LOSS iou_loss)
{
    ious all_ious = {0};
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);

    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);

    if (pred.w == 0)
        pred.w = 1.0;
    if (pred.h == 0)
        pred.h = 1.0;
    if (iou_loss == MSE)
    {
        float tx = truth.x * lw - i;
        float ty = truth.y * lh - j;
        float tw = log(truth.w * w / biases[2 * n]);
        float th = log(truth.h * h / biases[2 * n + 1]);

        delta[index + 0*stride] = class_weight * scale * (tx - x[index + 0*stride]);
        delta[index + 1*stride] = class_weight * scale * (ty - x[index + 1*stride]);
        delta[index + 2*stride] = class_weight * scale * (tw - x[index + 2*stride]);
        delta[index + 3*stride] = class_weight * scale * (th - x[index + 3*stride]);
    }
    else
    {
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        delta[index + 0 * stride] = all_ious.dx_iou.dt;
        delta[index + 1 * stride] = all_ious.dx_iou.db;
        delta[index + 2 * stride] = all_ious.dx_iou.dl;
        delta[index + 3 * stride] = all_ious.dx_iou.dr;

        delta[index + 2 * stride] *= exp(x[index + 2 * stride]);
        delta[index + 3 * stride] *= exp(x[index + 3 * stride]);

        delta[index + 0 * stride] *= (iou_normalizer * class_weight);
        delta[index + 1 * stride] *= (iou_normalizer * class_weight);
        delta[index + 2 * stride] *= (iou_normalizer * class_weight);
        delta[index + 3 * stride] *= (iou_normalizer * class_weight);
    }

    return all_ious;
}

void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat, float *class_weights)
{
    int n;
    if (delta[index])
    {
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) 
            *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n)
    {
        /* delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n]; */
        delta[index + stride * n] = ((n == class) ? class_weights[class] : 0) - output[index + stride * n];
        if(n == class && avg_cat) 
            *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    float *class_weights = net.class_weights;

    int len = 0;
    while(class_weights[len] != 0)
    {
        len++;
    }
    if (len != l.classes)
    {
        printf("the dimension of loss weights is not equal to the number of classse!\n");
        exit(0);
    }

    /* int idx; */
    /* for (idx = 0; idx < len; idx++) */
    /* { */   
    /*     printf("class_weights %d:%f\n", idx, loss_w[idx]); */
    /* } */

    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b)
    {
        for(n = 0; n < l.n; ++n)
        {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) 
        return;
    
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;

    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;

    /* float avg_iou = 0; */
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) 
    {
        for (j = 0; j < l.h; ++j) 
        {
            for (i = 0; i < l.w; ++i) 
            {
                for (n = 0; n < l.n; ++n) 
                {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t)
                    {
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) 
                        {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);
                    if (best_iou > l.ignore_thresh) 
                    {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) 
                    {
                        l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0, class_weights);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box_iouloss(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h,  class_weights[class], l.iou_normalizer, l.iou_loss);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t)
        {
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n)
            {
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0)
            {
                int class = net.truth[t*(4 + 1) + b*l.truths + 4];

                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                ious all_ious = delta_yolo_box_iouloss(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, class_weights[class], l.iou_normalizer, l.iou_loss);

                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;

                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);

                /* int class = net.truth[t*(4 + 1) + b*l.truths + 4]; */
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat, class_weights);

                ++count;
                ++class_count;
                if(all_ious.iou > .5) 
                    recall += 1;
                if(all_ious.iou > .75) 
                    recall75 += 1;
                /* avg_iou += iou; */
            }
        }
    }

    int stride = l.w*l.h;
    float* no_iou_loss_delta = calloc(l.batch * l.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) 
    {
        for (j = 0; j < l.h; ++j) 
        {
            for (i = 0; i < l.w; ++i) 
            {
                for (n = 0; n < l.n; ++n) 
                {
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    no_iou_loss_delta[index + 0*stride] = 0;
                    no_iou_loss_delta[index + 1*stride] = 0;
                    no_iou_loss_delta[index + 2*stride] = 0;
                    no_iou_loss_delta[index + 3*stride] = 0;
                }
            }
        }
    }
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);

    float avg_iou_loss = 0;
    if (l.iou_loss == MSE ) 
    {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    } 
    else if (l.iou_loss == IOU) 
    {
        avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
    } 
    else if (l.iou_loss == GIOU) 
    {
        avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
    } 
    else if (l.iou_loss == DIOU) 
    {
        avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_diou_loss / count) : 0;
    } 
    else 
    {
        avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_ciou_loss / count) : 0;
    } 
    *(l.cost) = avg_iou_loss + classification_loss;

    printf("v3 (%s loss, Normalizer: (iou: %f, cls: %f) Region %d Avg (IOU: %f, %s: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d\n", \
           (l.iou_loss==MSE?"mse":(l.iou_loss==IOU?"iou":(l.iou_loss==GIOU?"giou":(l.iou_loss==DIOU?"diou":"ciou")))), \
           l.iou_normalizer, \
           l.cls_normalizer, \
           net.index, \
           tot_iou/count, \
           (l.iou_loss==GIOU?"GIOU":(l.iou_loss==DIOU?"DIOU":"CIOU")), \
           (l.iou_loss==GIOU? tot_giou/count:(l.iou_loss==DIOU? tot_diou/count:tot_ciou/count)), \
           avg_cat/class_count, \
           avg_obj/count, \
           avg_anyobj/(l.w*l.h*l.n*l.batch), \
           recall/count, \
           recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n)
        {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) 
                continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j)
            {
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

