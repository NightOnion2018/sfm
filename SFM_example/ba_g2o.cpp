#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdint.h>
#include <Eigen/StdVector>
#include <map>


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

using namespace std;

struct free_delete
{
    void operator()(void* x) { free(x); }
};

/* 
class SFM{
    public:
        float focal_length = 0;
        float cx = 0;
        float cy = 0;
        int num_images = 0;
        int num_landmarks = 0;
}
*/
int main(int argc, char **argv){
    ifstream sfm_g2o;
    sfm_g2o.open(argv[1]);
    string line1;
    float fl, cx, cy;
    float x, y, z;
    int im_vertex_idx, landmark_vertex_idx;
    float u, v;
    int num_images, num_landmarks;

    // 0. load image intrincsic information
    sfm_g2o >> fl >> cx >> cy;
    std::getline(sfm_g2o, line1);
    sfm_g2o >> num_images;
    std::getline(sfm_g2o, line1);
    cout<<"Focal Length: " << fl << endl;
    cout<<"C_x: " << cx <<  "  C_y: " << cy << endl;
    cout<<"Num of Image "<<num_images<<endl;

    // 1. load landmark vertex information
    sfm_g2o >> num_landmarks;
    std::getline(sfm_g2o, line1);
    //cout<<"Num of Landmard "<<num_landmarks<<endl;
    float *landmarks = (float *)malloc(sizeof(float) * 3 * num_landmarks);
    //std::unique_ptr<float, free_delete> landmarks((float *)malloc(sizeof(float) * 3 * num_landmarks));
    for(int i = 0; i < num_landmarks; i++){
        sfm_g2o >>  x >> y >> z;
        landmarks[i * 3 + 0] = x;
        landmarks[i * 3 + 1] = y;
        landmarks[i * 3 + 2] = z;
        std::getline(sfm_g2o, line1);
    }

    // 2. load image_vertex <-> landmark_vertex as edges
    map<pair<int, int>, pair<float, float> > edges;
    for(int i = 0; i < num_images; i++){
        int num_image_landmark_pairs;
        sfm_g2o >> num_image_landmark_pairs;
        std::getline(sfm_g2o, line1);
        for(int j = 0; j < num_image_landmark_pairs; j++){
            sfm_g2o >> im_vertex_idx >> landmark_vertex_idx >> u >> v;
            std::getline(sfm_g2o, line1);
            pair<int, int> key = make_pair(im_vertex_idx, landmark_vertex_idx);
            pair<float, float> val = make_pair(u, v);
            edges.insert(make_pair(key, val));
        }
    }

    // 3. load image names
    string image_names[10];
    for(size_t i = 0; i < num_images; i++){
        std::getline(sfm_g2o, image_names[i]);
    }

    sfm_g2o.close();

    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    // add image poses as vertex
    for(int i = 0; i < num_images; i++){
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if(i == 0) v->setFixed(true);
        Eigen::Quaterniond q;
        Eigen::Vector3d trans(0, 0., 0);
        q.setIdentity();
        v->setEstimate(g2o::SE3Quat(q, trans));
        optimizer.addVertex(v);
        //cout<<"cam pos "<<i<<" "<<endl;
    }
    // add landmark as vertex
    for(int i=0; i < num_landmarks; i++){
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        x = landmarks[i * 3 + 0];
        y = landmarks[i * 3 + 1];
        z = landmarks[i * 3 + 2];
        v->setId(num_images + i);
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
        //cout<<"landmark "<<num_images + i<<": ("<<x<<","<<y<<","<<z<<")"<<endl;
    }
    // add camera parameters
    g2o::CameraParameters *camera = new g2o::CameraParameters(fl, Eigen::Vector2d(cx, cy), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // add edges between image pose and landmark
    for(auto const &item : edges){
        pair<int, int> key = pair<int, int>(item.first);
        pair<float, float> value = pair<float, float>(item.second);
        //cout<<"("<<key.first<<','<<key.second<<")"<<endl;
        im_vertex_idx = key.first;
        landmark_vertex_idx = key.second;
        u = value.first;
        v = value.second;

        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(landmark_vertex_idx)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(im_vertex_idx)));
        edge->setMeasurement(Eigen::Vector2d(u, v));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // start to optimize
    cout<< "Optimization Starting..." << endl;
    optimizer.setVerbose(true);
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    size_t num_iteration = 1000;
    if(argc == 3) num_iteration = atoi(argv[2]);
    cout<<"Run Optimization with "<< num_iteration << " Iterations"<<endl;
    optimizer.optimize(num_iteration);
    cout<<"Optimization Done."<<endl;

    // measure errors
    optimizer.computeActiveErrors();



    // output to PMVS for visualization the cloud point
    std::system("rm -rf root/visualize; mkdir root/visualize");
    std::system("rm -rf root/txt;       mkdir root/txt");

    std::ofstream option("root/options.txt");

    option << "timages  -1 " << 0 << " " << num_images - 1 << endl;
    option << "oimages 0" << endl;
    option << "level 1" << endl;
    option.close();
 

    g2o::ParameterContainer optimized_params = optimizer.parameters();
    g2o::CameraParameters *cam_optimized_K = dynamic_cast<g2o::CameraParameters *>
                                             (optimized_params.getParameter(camera->id()));
    number_t cam_optimized_fl = cam_optimized_K->focal_length;
    g2o::Vector2 cam_optimized_principle_point = cam_optimized_K->principle_point;
    cout<<"After Bundle Adjustment"<<endl;
    cout<<"    F_x            : "<<cam_optimized_fl<<endl;
    cout<<"    Principle Point:("<< cam_optimized_principle_point[0];
    cout<<","<< cam_optimized_principle_point[0]<< ")" <<endl;
    Eigen::MatrixXd K(3,3);
    K.setZero();
    K(0,0) = cam_optimized_fl;
    K(1,1) = cam_optimized_fl;
    K(0,2) = cam_optimized_principle_point[0];
    K(1,2) = cam_optimized_principle_point[1];
    K(2,2) = 1;

    float scale = 1.0;
    for(size_t i = 0; i < num_images; i++){
        char printbuf[256];
        // copy image to local folder
        sprintf(printbuf, "cp -f %s root/visualize/%04d.jpg", image_names[i].c_str(), (int)i);
        system(printbuf);

        // output image pose parameters to txt file
        sprintf(printbuf, "root/txt/%04d.txt", (int)i);
        ofstream out_txt(printbuf);

        g2o::VertexSE3Expmap *cam_i = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex((int)i));
        const g2o::SE3Quat& cam_i_estimate = const_cast<g2o::SE3Quat&>(cam_i->estimate());
        const g2o::Quaternion& cam_i_rotation = cam_i_estimate.rotation();
        const g2o::Vector3& cam_i_translation = cam_i_estimate.translation();
        Eigen::MatrixXd p_matrix(3, 4);
        Eigen::MatrixXd t(3,1);

        g2o::VertexSBAPointXYZ *landmark = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(30));
        g2o::Vector3 &xyz = const_cast<g2o::Vector3 &>(landmark->estimate());
        //t << (cam_i_rotation.matrix() * (cam_i_translation));
        //t = -t;
        //p_matrix << cam_i_rotation.matrix().transpose() , t;
        
        if(i == 1){
            scale = 50.0 / cam_i_translation(0,0);
            cout<<"Scale is "<<scale<<endl;
        }
        t << (cam_i_translation)*scale;
        p_matrix << cam_i_rotation.matrix() , t;
        
        Eigen::MatrixXd P(3, 4);
        P << (K * p_matrix);
        cout<<"p_matrix Matrix["<<i<<"]:"<<endl;
        cout<<p_matrix<<endl;
        out_txt << "CONTOUR" << endl;
        for (int j=0; j < 3; j++) {
            for (int k=0; k < 4; k++) {
                out_txt << P(j, k) << " ";
            }
            out_txt << endl;
        }
        out_txt.close();
    }

    // dummy code below
}
