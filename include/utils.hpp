#ifndef LIDAR2OSM_TESTS_UTILS_H_
#define LIDAR2OSM_TESTS_UTILS_H_

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

#include "kitti_conversions.hpp"

bool readBin(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>& cloud) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return false;
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_points            = points_bytes / (sizeof(Eigen::Vector4f));

  ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector4f> points(num_points);
  ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector4f) * num_points);

  cloud.clear();
  cloud.reserve(num_points);

  pcl::PointXYZ point;
  for (auto& pt : points) {
    point.x = pt(0);
    point.y = pt(1);
    point.z = pt(2);
    cloud.emplace_back(point);
  }

  return true;
}

Eigen::MatrixXd transformEigenCloud(const Eigen::MatrixXd& input_cloud,
  const Eigen::Affine3d& tf) {
  Eigen::MatrixXd output_cloud = input_cloud;

  for (int i = 0; i < output_cloud.rows(); ++i) {
    Eigen::Vector3d pt(output_cloud(i, 0), output_cloud(i, 1), output_cloud(i, 2));
    Eigen::Vector3d pt_tf = tf * pt;

    output_cloud(i, 0) = pt_tf.x();
    output_cloud(i, 1) = pt_tf.y();
    output_cloud(i, 2) = pt_tf.z();
  }

  return output_cloud;
}

bool loadPointCloudRGB(const std::string& filepath, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
  std::string extension = std::filesystem::path(filepath).extension().string();

  if (extension == ".pcd") {
    return pcl::io::loadPCDFile<pcl::PointXYZRGB>(filepath, *cloud) >= 0;
  } else {
    std::cerr << "Unsupported file format: " << extension << std::endl;
    return false;
  }
}

bool loadPointCloud(const std::string& filepath, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  std::string extension = std::filesystem::path(filepath).extension().string();

  if (extension == ".pcd") {
    return pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *cloud) >= 0;
  } else if (extension == ".ply") {
    return pcl::io::loadPLYFile<pcl::PointXYZ>(filepath, *cloud) >= 0;
  } else if (extension == ".bin") {
    return readBin(filepath, *cloud);
  } else {
    std::cerr << "Unsupported file format: " << extension << std::endl;
    return false;
  }
}

std::pair<double,double> calcRRE(Eigen::Affine3d A_est, Eigen::Affine3d A_gt) {
  // 1) Extract the rotation matrices and translation vectors
  Eigen::Affine3d A_delta = A_gt * A_est;
  Eigen::Matrix3d R_delta = A_delta.linear();
  Eigen::Vector3d t_delta = A_delta.translation();

  // 2) Compute the difference of trans and find Euclidean (L2) translational error
  double trans_err = t_delta.norm();
  // double err_x = std::abs(t_delta.x());
  // double err_y = std::abs(t_delta.y());
  // double err_z = std::abs(t_delta.z());

  // 3) Clamp trace for robustness, then compute the angle
  double tr = R_delta.trace();
  tr = std::min(3.0, std::max(-1.0, tr));
  double rot_err_rad = std::acos((tr - 1.0) / 2.0);
  double rot_err_deg = rot_err_rad * 180.0 / M_PI;

  return { trans_err, rot_err_deg };
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertEigenToCloud(const Eigen::MatrixXd& eigen_cloud) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl_cloud->points.reserve(eigen_cloud.rows());
  
  for (int i = 0; i < eigen_cloud.rows(); ++i) {
    pcl::PointXYZRGB pt;
    pt.x = eigen_cloud(i, 0);
    pt.y = eigen_cloud(i, 1);
    pt.z = eigen_cloud(i, 2);

    int label = static_cast<int>(eigen_cloud(i, 3));
    std::vector<int> rgb = getRGBFromLabel(label);

    if (rgb[0] >= 0) {  // Check for valid label
        uint32_t rgb_packed = (static_cast<uint32_t>(rgb[0]) << 16 |
                                static_cast<uint32_t>(rgb[1]) << 8 |
                                static_cast<uint32_t>(rgb[2]));
        pt.rgb = *reinterpret_cast<float*>(&rgb_packed);
    } else {
        pt.r = pt.g = pt.b = 0;  // fallback to black
    }

    pcl_cloud->points.push_back(pt);
  }
  
  pcl_cloud->width = pcl_cloud->points.size();
  pcl_cloud->height = 1;
  pcl_cloud->is_dense = true;
  
  return pcl_cloud;
}

Eigen::MatrixXd convertCloudToEigen(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud, int max_num_points) {
  size_t original_size = pcl_cloud->size();
  size_t target_size = std::min<size_t>(max_num_points, original_size);  // Limit to max_num_points

  // Generate indices and randomly shuffle
  std::vector<size_t> indices(original_size);
  std::iota(indices.begin(), indices.end(), 0);     // Fill with 0,1,2,...,original_size-1
  std::random_device rd;
  std::mt19937 g(rd());                             // Random seed
  std::shuffle(indices.begin(), indices.end(), g);
  
  // convert the PCL points to an Eigen Matrix (XYZ + Label)
  Eigen::MatrixXd eigen_cloud(target_size, 4);
  for (size_t i = 0; i < target_size; ++i) {
    size_t idx = indices[i];  // Pick a random index
    eigen_cloud(i, 0) = pcl_cloud->points[idx].x;
    eigen_cloud(i, 1) = pcl_cloud->points[idx].y;
    eigen_cloud(i, 2) = pcl_cloud->points[idx].z;
    // Extract RGB values
    uint32_t rgb_val = *reinterpret_cast<int*>(&pcl_cloud->points[idx].rgb);
    int r = (rgb_val >> 16) & 0x0000ff;
    int g = (rgb_val >> 8) & 0x0000ff;
    int b = (rgb_val) & 0x0000ff;
    eigen_cloud(i, 3) = getLabelFromRGB(r, g, b);  // Assign label
  }

  std::cout << "Converted global map to Eigen matrix with " <<  target_size << 
    " points (downsampled from " << original_size << ")\n"; 

  return(eigen_cloud);
}

Eigen::MatrixXd convertVecToEigen(
    const std::vector<Eigen::Vector3f>& pts)
    // const std::vector<int>& labels)
{
    // assert(pts.size() == labels.size());
    const size_t N = pts.size();
    Eigen::MatrixXd M(N, 4);

    for (size_t i = 0; i < N; ++i) {
        M(i, 0) = pts[i].x();
        M(i, 1) = pts[i].y();
        M(i, 2) = pts[i].z();
        M(i, 3) = static_cast<double>(0);
        // M(i, 3) = static_cast<double>(labels[i]);
    }
    return M;
}

std::vector<Eigen::Vector3f> convertCloudToVec(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  std::vector<Eigen::Vector3f> vec;
  vec.reserve(cloud.size());
  for (const auto& pt : cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    vec.emplace_back(pt.x, pt.y, pt.z);
  }
  return vec;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr convertVecToCloud(std::vector<Eigen::Vector3f> vectorPC) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPC(new pcl::PointCloud<pcl::PointXYZ>);
  cloudPC->points.reserve(vectorPC.size());

  // Copy each Eigen::Vector3f into a PointXYZ
  for (const auto &v : vectorPC) {
      pcl::PointXYZ p;
      p.x = v.x();
      p.y = v.y();
      p.z = v.z();
      cloudPC->points.push_back(p);
  }

  return cloudPC;
}

Eigen::Matrix4f getRandTF(float minTrans, float maxTrans, float minRot, float maxRot) {
  // ——— set up RNG and distributions ———
  std::random_device rd;
  std::mt19937 gen(rd());

  // translation in [0,100]
  std::uniform_real_distribution<float> dist_trans(minTrans, maxTrans);

  // rotation angle in [45°,180°] (in radians)
  std::uniform_real_distribution<float> dist_angle(minRot * M_PI / 180.0f, maxRot * M_PI / 180.0f);

  // for random axis: sample each coord from N(0,1) then normalize
  std::normal_distribution<float> dist_norm(0.0f, 1.0f);

  // ——— sample translation ———
  Eigen::Vector3f t;
  t << dist_trans(gen),
       dist_trans(gen),
       dist_trans(gen);

  // ——— sample random axis ———
  Eigen::Vector3f axis(dist_norm(gen), dist_norm(gen), dist_norm(gen));
  axis.normalize();  // make it unit length

  // ——— sample random angle ———
  float θ = dist_angle(gen);

  // ——— build rotation matrix ———
  Eigen::AngleAxisf aa(θ, axis);
  Eigen::Matrix3f R = aa.toRotationMatrix();

  // ——— assemble the 4×4 transform ———
  Eigen::Matrix4f X = Eigen::Matrix4f::Identity();
  X.block<3,3>(0,0) = R;
  X.block<3,1>(0,3) = t;

  return X;
}

void colorize(pcl::PointCloud<pcl::PointXYZRGB> &pc_colored, const std::vector<int> &color) {
  int N = pc_colored.points.size();
  for (int i = 0; i < N; ++i) {
    pc_colored.points[i].r       = color[0];
    pc_colored.points[i].g       = color[1];
    pc_colored.points[i].b       = color[2];
  }
}

void colorize(const pcl::PointCloud<pcl::PointXYZ> &pc,
              pcl::PointCloud<pcl::PointXYZRGB> &pc_colored,
              const std::vector<int> &color) {
  int N = pc.points.size();

  pc_colored.clear();
  pcl::PointXYZRGB pt_tmp;
  for (int i = 0; i < N; ++i) {
    const auto &pt = pc.points[i];
    pt_tmp.x       = pt.x;
    pt_tmp.y       = pt.y;
    pt_tmp.z       = pt.z;
    pt_tmp.r       = color[0];
    pt_tmp.g       = color[1];
    pt_tmp.b       = color[2];
    pc_colored.points.emplace_back(pt_tmp);
  }
}

#endif  // LIDAR2OSM_TESTS_UTILS_H_