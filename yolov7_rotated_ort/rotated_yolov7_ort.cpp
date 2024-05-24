#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <Eigen/Dense>
#include "time.h"
using namespace cv;
using namespace std;
int model_input_width;
int model_input_height;

float conf_thres = 0.5;
float iou_thres = 0.45;
int nms_top_k_=5;
int num_class = 2;
float polygoniou(std::vector<std::vector<float>>& pointArray1, std::vector<std::vector<float>>& pointArray2);
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
typedef struct Rotated_Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    int label;
    float prob;
    Rotated_Detection() {}
    friend bool operator>(const Rotated_Detection &lhs, const Rotated_Detection &rhs) {
    return (lhs.prob > rhs.prob);
  }
     ~Rotated_Detection() {}
} Rotated_Detection;


float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

//前处理
cv::Mat letterbox(cv::Mat &src, int h, int w, std::vector<float> &padding)
{
    // Resize and pad image while meeting stride-multiple constraints
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);//round四舍五入
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;

    // resize
    resize(src, resize_img, cv::Size(inside_w, inside_h));

    // divide padding into 2 sides
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padding.push_back(padd_w);
    padding.push_back(padd_h);

    // store the ratio
    padding.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));

    // add border
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

/*计算边界框在原始图像上的坐标,限制边界框坐标在图像范围内*/
void polygon_scale_coords(std::vector<int> img1_shape, 
Rotated_Detection &coords, std::vector<int> img0_shape) {
    // Calculate gain and padding
    float gain, pad_x, pad_y;
    gain = std::min(static_cast<float>(img1_shape[0]) / img0_shape[0], static_cast<float>(img1_shape[1]) / img0_shape[1]);
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2;
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2;
    // Adjust coordinates
        coords.x1 -= pad_x;  // Adjust x coordinates
        coords.x2 -= pad_x;
        coords.x3 -= pad_x;
        coords.x4 -= pad_x;
        coords.y1 -= pad_y;  // Adjust y coordinates
        coords.y2 -= pad_y;
        coords.y3 -= pad_y;
        coords.y4 -= pad_y;
        coords.x1 /= gain;  // Scale coordinates
        coords.x2 /= gain; 
        coords.x3 /= gain; 
        coords.x4 /= gain; 
        coords.y1 /= gain; 
        coords.y2 /= gain; 
        coords.y3 /= gain; 
        coords.y4 /= gain; 
        coords.x1 = std::max(0.0f, std::min(coords.x1, static_cast<float>(img0_shape[1])));
        coords.x2 = std::max(0.0f, std::min(coords.x2, static_cast<float>(img0_shape[1])));
        coords.x3 = std::max(0.0f, std::min(coords.x3, static_cast<float>(img0_shape[1])));
        coords.x4 = std::max(0.0f, std::min(coords.x4, static_cast<float>(img0_shape[1])));
        coords.y1 = std::max(0.0f, std::min(coords.y1, static_cast<float>(img0_shape[0])));
        coords.y2 = std::max(0.0f, std::min(coords.y2, static_cast<float>(img0_shape[0])));
        coords.y3 = std::max(0.0f, std::min(coords.y3, static_cast<float>(img0_shape[0])));
        coords.y4 = std::max(0.0f, std::min(coords.y4, static_cast<float>(img0_shape[0])));
}
// 画框
void polygon_plot_one_box(const Rotated_Detection &coords, cv::Mat& im) {
    const cv::Scalar color = cv::Scalar(128, 128, 128);
    int line_thickness = 3;
    int tl = line_thickness > 0 ? line_thickness : round(0.002 * (im.rows + im.cols) / 2) + 1;
    float fenge_distance_x = coords.x2 - coords.x1;
    float fenge_distance_y = coords.y2 - coords.y1;
    float jinru_distance = sqrt(pow(coords.x1 - coords.x4, 2) + pow(coords.y1 - coords.y4, 2));
    int length = jinru_distance > 240 ? 130 : 300;
    int H = im.rows, W = im.cols;
    float x_ratio = 1, y_ratio = (float)H / W;
    float radian = atan2(fenge_distance_y * y_ratio, fenge_distance_x * x_ratio);
    float sep_cos = cos(radian);
    float sep_sin = sin(radian);
    int p3_x = cvRound(coords.x1 + length * sep_cos);
    int p3_y = cvRound(coords.y1 + length * sep_sin);
    int p2_x = cvRound(coords.x4 + length * sep_cos);
    int p2_y = cvRound(coords.y4 + length * sep_sin);
    vector<Point> points = {
        Point(cvRound(coords.x1), cvRound(coords.y1)),
        Point(p3_x, p3_y),
        Point(p2_x, p2_y),
        Point(cvRound(coords.x4), cvRound(coords.y4))
    };
    vector<vector<Point>> contours = { points };
    polylines(im, contours, true, color, tl, LINE_AA);
    std::stringstream text_ss;
    std::string class_name ;
    if(coords.label)
    {class_name="no_park";}
    else {class_name="park";}
      text_ss << class_name << ":" << std::fixed
              << std::setprecision(4) << coords.prob;
      cv::putText(
          im,
          text_ss.str(),
          cv::Point(coords.x1+5, coords.y1+20),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          cv::Scalar(0, 0, 255),
          1,
          cv::LINE_AA);
    

}
void yolo7_rotated_nms(std::vector<Rotated_Detection> &input,
               float iou_threshold,
               int top_k,
               std::vector<Rotated_Detection> &result,
               bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Rotated_Detection>());
  std::vector<bool> skip(input.size(), false);
  int count = 0;//topk=5 ,skip.size()=input.size()
  for (size_t i = 0; count < top_k && i < skip.size(); i++) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;
    //和选中的框进行遍历对比
    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      if (suppress == false) {
        if (input[i].label != input[j].label) {
          continue;
        }
      }
      std::vector<std::vector<float>> box={{input[i].x1,input[i].y1},{input[i].x2,input[i].y2},{input[i].x3,input[i].y3},{input[i].x4,input[i].y4}};
      std::vector<std::vector<float>> box1={{input[j].x1,input[j].y1},{input[j].x2,input[j].y2},{input[j].x3,input[j].y3},{input[j].x4,input[j].y4}};
      float res_iou = polygoniou(box,box1);
        if (res_iou > iou_threshold) {
          skip[j] = true;
        }
    }
    result.push_back(input[i]);
  }
}

void ort_process(const std::string& onnx_path_name, cv::Mat& image,std::vector<Rotated_Detection> &Dets) {
    vector<float> input_image_;
    int row = image.rows;
    int col = image.cols;
    input_image_.resize(row * col * image.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = image.ptr<uchar>(i)[j * 3 + 2 - c];
                //input_image_[c * row * col + i * col + j] = (pix / 255.0 - mean_vals[c]) / norm_vals[c];
                input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, onnx_path_name.c_str(), session_options);

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    int output_h = 0;
	int output_w = 0;
for(int i=0;i<num_output_nodes;i++){
	Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[2]; 
	output_w = output_dims[3]; 
	std::cout << "output format : HxW = " << output_dims[2] << "x" << output_dims[3] << std::endl;
}                       
    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        input_node_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    }

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        output_node_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    }

    for (auto input_name : input_node_names)
    {
        std::cout << "input node name   : " << input_name << std::endl;
    }

    for (auto output_name : output_node_names)
    {
        std::cout << "output node name  : " << output_name << std::endl;
    }
    std::cout << std::endl;
    std::vector<int64_t> inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    model_input_height = inputDims.at(3);
    model_input_width = inputDims.at(2);
    std::cout << "model_input_height : " << model_input_height<< " model_input_width : " << model_input_width << std::endl;
    vector<int32_t> strides={8,16,32};
std::vector<std::vector<std::vector<GridAndStride>>>points_;
if (points_.empty()) {
    points_.resize(3);
    for (int i = 0; i < 3; i++) {
        int stride = strides[i];
        int num_grid_w = model_input_width / stride;
        int num_grid_h = model_input_height / stride;
        points_[i].resize(num_grid_w);
      std::vector<std::vector<GridAndStride>>&ps = points_[i];
       for (int g1 = 0; g1 < num_grid_w; g1++)//96行32列
        {   
            points_[i][g1].resize(num_grid_h);
            for (int g0 = 0; g0 < num_grid_h; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                points_[i][g1][g0]=gs;//points_[0][3][8]:grid0=8,grid1=3
            }
        }
    }
   }
   /*三维数组points的大小 points[0]：96*32 points[1]：48*16 points[2]:24*8*/
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
        input_image_.data(), 
        input_image_.size(),
        inputDims.data(),
        inputDims.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 3> outNames = { output_node_names[0].c_str(),output_node_names[1].c_str(),output_node_names[2].c_str() };
     auto start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr },
        inputNames.data(),
        &inputTensor,
        num_input_nodes,
        outNames.data(),
        num_output_nodes); 
    
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // 输出执行时间
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    // 测试推理输出结果
    /*float* pdata=outputTensors[0].GetTensorMutableData<float>();
    for (int i=8;i<200;i=i+11)
    {cout<<"conf："<<*(pdata+i)<<"  cls1："<<*(pdata+i+1)<<"  cls2："<<*(pdata+i+2)<<endl;}
    //(pdata+i)=pdata[i]
    float* testaddr=pdata+96*32*11*3;
    for (int i=8;i<200;i=i+11)
    {cout<<"conf："<<*(testaddr+i)<<"  cls1："<<*(testaddr+i+1)<<"  cls2："<<*(testaddr+i+2)<<endl;}
    float* pdata1=outputTensors[1].GetTensorMutableData<float>();
    cout<<"pdata:"<<pdata<<"  pdata1:"<<pdata1<<endl;
    std::vector<int64_t> outputShape = outputTensors[2].GetTensorTypeAndShapeInfo().GetShape();//{1，3，96，32，11}
    for(int i=0;i<outputShape.size();i++)
    {cout<<"  "<<outputShape[i]<<endl;}
    size_t count = outputTensors[num].GetTensorTypeAndShapeInfo().GetElementCount();//101376=1*3*96*32*11
    float* pdata0=pdata;*/
for(int num=0;num<3;num++)//3个输出头
{
    float* pdata0=outputTensors[num].GetTensorMutableData<float>();
    cout<<"pdata0:"<< pdata0<<" "<<"pdata0[8]"<<pdata0[8]<<endl;
     int stride=strides[num];
     int num_w = model_input_width / stride;
     int num_h = model_input_height / stride;
    /*取出网格数据*/
   for (int anchors = 0; anchors < 3; anchors++)//anchor数量
          {    
    for (int w=0; w<num_w; w++ ){
        for (int h=0; h<num_h; h++){
        float obj_conf = sigmoid(pdata0[8]);
        /*测试 12096*/
        int id = std::distance(pdata0+9, std::max_element(pdata0+9, pdata0+9+num_class));
        float box_cls_score = sigmoid(pdata0[9+id]);
        float box_prob = obj_conf * box_cls_score;
        if (box_prob > conf_thres) {
        int grid0 = points_[num][w][h].grid0;
        int grid1 = points_[num][w][h].grid1;
        int stride = points_[num][w][h].stride;
        float x1 = (pdata0[0] + grid0) * stride;
        float y1 = (pdata0[1] + grid1) * stride;
        float x2 = (pdata0[2] + grid0) * stride;
        float y2 = (pdata0[3] + grid1) * stride;
        float x3 = (pdata0[4] + grid0) * stride;
        float y3 = (pdata0[5] + grid1) * stride;
        float x4 = (pdata0[6] + grid0) * stride;
        float y4 = (pdata0[7] + grid1) * stride;
        Rotated_Detection Det;
            Det.x1 = x1;
            Det.y1 = y1;
            Det.x2 = x2;
            Det.y2 = y2;
            Det.x3 = x3;
            Det.y3 = y3;
            Det.x4 = x4;
            Det.y4 = y4;
            Det.label = id;
            Det.prob = box_prob;
            Dets.push_back(Det);}
        pdata0 += 11;
        //cout<<"pdata0："<<pdata0<<endl;//float是四个字节，字节变化4*11
            }
        }
    }
  }
    session_options.release();
    session.release();   
}
/*iou计算*/
float crossProduct(float x1, float y1, float x2, float y2) {
        return x1 * y2 - x2 * y1;
    }
std::vector<std::vector<float>> sortPoints(std::vector<std::vector<float>> &pointArray) {
        std::vector<std::vector<float>> result;
        //cout<<"pointArray.size():"<<pointArray.size()<<endl;
        if (pointArray.size() < 3) {
            //std::cerr << "no points need sort" << std::endl;
            return pointArray;
        }

        if (pointArray.size() > 10000) {
            std::cerr << "too many data" << std::endl;
            return pointArray;
        }
        // 查找x最小
        float xMin = pointArray[0][0];
        int iMin = 0;
        for (int i = 0; i < pointArray.size(); ++i) {
            if (pointArray[i][0] < xMin) {
                xMin = pointArray[i][0];
                iMin = i;
            }
        }
        // 开始点
        float x0 = pointArray[iMin][0];
        float y0 = pointArray[iMin][1];
        result.push_back({x0, y0});
        // 排序
        for (int i = 0; i < pointArray.size(); ++i) {
            if (i == iMin) {  // 跳过原点
                continue;
            }
            float x1 = pointArray[i][0] - x0;
            float y1 = pointArray[i][1] - y0;

            for (int j = i + 1; j < pointArray.size(); ++j) {
                if (j == iMin) {  // 跳过原点
                    continue;
                }
                float x2 = pointArray[j][0] - x0;
                float y2 = pointArray[j][1] - y0;

                float PxQ = crossProduct(x1, y1, x2, y2);
                // 两点对调。判断标准，逆时针，如果同轴，判断距离原点更远
                if (PxQ > 0 || (0 == PxQ && (fabs(x1) + fabs(y1) > fabs(x2) + fabs(y2)))) {
                    // 对调
                    std::swap(pointArray[j], pointArray[i]);
                    x1 = x2;
                    y1 = y2;
                }
            }
            result.push_back({pointArray[i][0], pointArray[i][1]});
        }
        return result;
    }
// 判断是否是凸多边形
 bool isConvexPolygon(std::vector<std::vector<float>>& pointArray) {
    if (pointArray.size() < 4) {
          return true;
    }
    for (size_t i = 0; i < pointArray.size() - 1; ++i) {  // 排序后0点一定是凸点，不用判断
        float x1 = pointArray[i + 1][0] - pointArray[i][0];
        float y1 = pointArray[i + 1][1] - pointArray[i][1];
        float x2 = pointArray[(i + 2) % pointArray.size()][0] - pointArray[i + 1][0];
        float y2 = pointArray[(i + 2) % pointArray.size()][1] - pointArray[i + 1][1];
        float PxQ = crossProduct(x1, y1, x2, y2);
        if (PxQ > 0) {
            return false;
        }
    }
    return true;
}
 // 将凹多边形转换为凸多边形，凹的点会被移除
std::vector<std::vector<float>> convexToConverx(std::vector<std::vector<float>>& pointArray) {
    if (pointArray.size() < 4) {
        return pointArray;
    }
    bool needAgain = true;
    int round = 1;
    // 一轮可能无法全部去除，需要多轮
    while (needAgain && pointArray.size() > 3 && round < pointArray.size()) {
        needAgain = false;
        int i = 0;
        while (i < pointArray.size() - 1) {  // 排序后0点一定是凸点，不用判断
            float x1 = pointArray[i + 1][0] - pointArray[i][0];
            float y1 = pointArray[i + 1][1] - pointArray[i][1];
            float x2 = pointArray[(i + 2) % pointArray.size()][0] - pointArray[i + 1][0];
            float y2 = pointArray[(i + 2) % pointArray.size()][1] - pointArray[i + 1][1];
            float PxQ = crossProduct(x1, y1, x2, y2);

            if (PxQ >= 0) {  // 移除点
                pointArray.erase(pointArray.begin() + i + 1);

                needAgain = true;
                continue;
            }
            ++i;
        }
        ++round;
    }
    return pointArray;
}
// 判断点是否在多边形内部
bool isPointInPolygon(const std::vector<float>& point, std::vector<std::vector<float>>& pointArray) {
    if (pointArray.size() < 3) {
        return false;
    }

    for (size_t i = 0; i < pointArray.size(); ++i) {
        float x1 = pointArray[i][0] - point[0];
        float y1 = pointArray[i][1] - point[1];
        float x2 = pointArray[(i + 1) % pointArray.size()][0] - point[0];
        float y2 = pointArray[(i + 1) % pointArray.size()][1] - point[1];
        float PxQ = crossProduct(x1, y1, x2, y2);

        if (PxQ > 0) {  // ==0时候表示在某一条边上，视为内部
            return false;
        }
    }
    return true;
}
// 判断两条线段是否相交
bool isLineAcross(float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
    float x_ac = cx - ax;
    float y_ac = cy - ay;

    float x_ad = dx - ax;
    float y_ad = dy - ay;

    float x_ab = bx - ax;
    float y_ab = by - ay;

    float ACxAB = crossProduct(x_ac, y_ac, x_ab, y_ab);
    float ADxAB = crossProduct(x_ad, y_ad, x_ab, y_ab);

    if (ACxAB * ADxAB >= 0) {  // 方向相反，不交叉
        return false;
    }
    float x_ca = ax - cx;
    float y_ca = ay - cy;

    float x_cd = dx - cx;
    float y_cd = dy - cy;

    float x_cb = bx - cx;
    float y_cb = by - cy;

    float CAxCD = crossProduct(x_ca, y_ca, x_cd, y_cd);
    float CBxCD = crossProduct(x_cb, y_cb, x_cd, y_cd);
    if (CAxCD * CBxCD >= 0) {  // 方向相反，不交叉
        return false;
    }

    return true;
}
// 计算两线段的交点
std::vector<float> crossPoint(float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
    float x_ac = cx - ax;
    float y_ac = cy - ay;

    float x_ad = dx - ax;
    float y_ad = dy - ay;

    float x_ab = bx - ax;
    float y_ab = by - ay;

    float x_cd = dx - cx;
    float y_cd = dy - cy;

    float ACxAD = crossProduct(x_ac, y_ac, x_ad, y_ad);
    float ABxCD = crossProduct(x_ab, y_ab, x_cd, y_cd);

    assert(ABxCD != 0 && "Lines do not intersect"); // 断言确保两线段相交

    float rate = std::abs(ACxAD / ABxCD);

    float ox = ax + rate * (bx - ax);
    float oy = ay + rate * (by - ay);

    return {ox, oy};
}
// 计算凸多边形面积
float polygonArea(std::vector<std::vector<float>>& pointArray) {
    if (pointArray.size() < 3) {
        return 0.0;
    }

    float area = 0.0;

    for (size_t i = 0; i < pointArray.size() - 2; ++i) {
        float x1 = pointArray[i + 1][0] - pointArray[0][0];
        float y1 = pointArray[i + 1][1] - pointArray[0][1];
        float x2 = pointArray[i + 2][0] - pointArray[0][0];
        float y2 = pointArray[i + 2][1] - pointArray[0][1];
        float PxQ = crossProduct(x1, y1, x2, y2);
        area += 0.5 * std::abs(PxQ);
    }
    return area;
}
// 计算两个凸多边形的交叉面积
std::vector<std::vector<float>> twoPolygonCrossArea(std::vector<std::vector<float>>& pointArray1, std::vector<std::vector<float>>& pointArray2) {
    std::vector<std::vector<float>> pList;
    if (pointArray1.size() < 3 || pointArray2.size() < 3) {
        return pList;
    }

    // 在对方内部的点
    for (const auto& p : pointArray2) {
        if (isPointInPolygon(p, pointArray1)) {
            pList.push_back(p);
        }
    }

    for (const auto& p : pointArray1) {
        if (isPointInPolygon(p, pointArray2)) {
            pList.push_back(p);
        }
    }

    // 交叉点
    for (size_t i = 0; i < pointArray1.size(); ++i) {
        float ax = pointArray1[i][0];
        float ay = pointArray1[i][1];
        float bx = pointArray1[(i + 1) % pointArray1.size()][0];
        float by = pointArray1[(i + 1) % pointArray1.size()][1];
        for (size_t k = 0; k < pointArray2.size(); ++k) {
            float cx = pointArray2[k][0];
            float cy = pointArray2[k][1];
            float dx = pointArray2[(k + 1) % pointArray2.size()][0];
            float dy = pointArray2[(k + 1) % pointArray2.size()][1];
            if (isLineAcross(ax, ay, bx, by, cx, cy, dx, dy)) {  // 相交，求交点
                pList.push_back(crossPoint(ax, ay, bx, by, cx, cy, dx, dy));
            }
        }
    }
    // TODO: 重新排序
    pList=sortPoints(pList);
    return pList;
}
// 计算 IOU
float polygoniou(std::vector<std::vector<float>>& pointArray1, std::vector<std::vector<float>>& pointArray2) {
    if (pointArray1.empty() || pointArray2.empty() || pointArray1.size() < 3 || pointArray2.size() < 3) {
        std::cout << "Not polygons" << std::endl;
        return 0.0;
    }
    // 排序和转换成凸多边形
     pointArray1=sortPoints(pointArray1);
     pointArray2=sortPoints(pointArray2);
    //如果是凹多边形，将其转换为凸多边形
     //pointArray1 = convexToConverx(pointArray1);
     //pointArray2 = convexToConverx(pointArray2);
    // 计算面积
    float area1 = polygonArea(pointArray1);
    float area2 = polygonArea(pointArray2);
    std::vector<std::vector<float>> crossPoint = twoPolygonCrossArea(pointArray1, pointArray2);
    float crossArea = polygonArea(crossPoint);
    float iou = crossArea / (area1 + area2 - crossArea);
    return iou;
}
/*iou计算结束*/
int main(int argc, char *argv[]) {
    // set the hyperparameters
    int img_h = 768;
    int img_w = 256;
    int img_c = 3;
    int img_size = img_h * img_h * img_c;
    const float prob_threshold = 0.30f;
    const float nms_threshold = 0.60f;
    const std::string model_path="../yolov7.onnx";
    const char *image_path="../yolov7images/180824-150206CAM3_000015.jpg";
    const std::string device_name="CPU";
    //读图
    cv::Mat src_img = cv::imread(image_path);//768 x 256 x 3
    std::vector<float> padding;
    cout<<"前处理开始"<<endl;
    cv::Mat image = letterbox(src_img, img_h, img_w, padding);
    cout<<"ort准备"<<endl;
    std::vector<Rotated_Detection> Dets;
    ort_process(model_path,image,Dets);
    cout<<"dets维度:"<<Dets.size()<<endl;
    std::vector<Rotated_Detection> det_result;
    yolo7_rotated_nms(Dets, iou_thres, nms_top_k_,det_result, false);
    cout<<"det_result维度:"<<det_result.size()<<endl;
    std::string save_path = "../yolov7images/img1.jpg";  // img.jpg
    for (size_t i = 0; i < det_result.size(); ++i) {
       polygon_scale_coords({src_img.rows,src_img.cols},det_result[i],{768,256});
       polygon_plot_one_box(det_result[i], src_img);
    }
    /*imshow("Rotated Box", src_img);
    waitKey(0);
    destroyAllWindows();*/
    imwrite(save_path,src_img);
    return 0;
}

