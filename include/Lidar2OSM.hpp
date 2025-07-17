#pragma once

#include <filesystem>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

#include <random>     // For random sampling
#include <algorithm>  // For std::shuffle, std::sample
#include <tuple>      // for filterSemanticAssociations method

// Eigen :D
#include <Eigen/Dense>
#include <Eigen/Geometry>

// CLIPPER
#include <clipper/clipper.h>
#include <clipper/utils.h>

namespace lidar2osm {

struct Lidar2OSMConfig {
  double min_dist_threshold_ = 3.0;
  double eff_comms_dist_threshold_ = 20.0;
  double map_voxel_size_ = 1.0;
  int max_num_points_ = 10000;
  int max_associations_ = 10000;

  Lidar2OSMConfig(const double min_dist_threshold = 3.0,
                  const double eff_comms_dist_threshold = 20.0,
                  const double map_voxel_size = 1.0,
                  const int max_num_points = 10000,
                  const int max_associations = 20000) {

    if (map_voxel_size < 5e-3) {
      throw std::runtime_error(
          "Too small voxel size has been given. Please check your voxel size.");
    }

    min_dist_threshold_ = min_dist_threshold;
    eff_comms_dist_threshold_ = eff_comms_dist_threshold;
    map_voxel_size_ = map_voxel_size;
    max_num_points_ = max_num_points;
    max_associations_ = max_associations;
  }
};

// All-to-All Association matrix
clipper::Association get_a2a_assoc_matrix(int N1, int N2) {
  clipper::Association assoc_matrix(N1 * N2, 2);
  int i = 0;
  for (int n1 = 0; n1 < N1; ++n1) {
    for (int n2 = 0; n2 < N2; ++n2) {
        assoc_matrix(i, 0) = n1;
        assoc_matrix(i, 1) = n2;
        ++i;
    }
  }
  return assoc_matrix;
}

// Ego dist filterer
clipper::Association filter_by_ego_distance(
  const Eigen::MatrixXd& pc1, 
  const Eigen::MatrixXd& pc2, 
  const clipper::Association& A,
  const double eff_comms_dist_threshold) {
  std::vector<Eigen::Vector2i> valid_rows;
  
  for (int i = 0; i < A.rows(); ++i) {
    int idx1 = A(i, 0);
    int idx2 = A(i, 1);

    float dist1 = pc1.row(idx1).norm();
    float dist2 = pc2.row(idx2).norm();

    float relative_ego_dist = std::abs(dist1 - dist2);

    if (relative_ego_dist < eff_comms_dist_threshold) {
        valid_rows.emplace_back(idx1, idx2);
    }
  }
  
  // Create new association mat
  clipper::Association Anew(valid_rows.size(), 2);
  for (size_t i = 0; i < valid_rows.size(); ++i) {
    Anew(i, 0) = valid_rows[i][0];
    Anew(i, 1) = valid_rows[i][1];
  }
  return Anew;
}

// Semantic Filter
std::tuple<clipper::Association, std::vector<int> > filterSemanticAssociations(
  const std::vector<int> & labels1, 
  const std::vector<int> & labels2, 
  const clipper::Association& A) {
  std::vector<Eigen::RowVector2i> filtered_rows;
  std::vector<int>  filteredLabels;

  for (int i = 0; i < A.rows(); ++i) {
    // Fetch indices from the association matrix
    int index1 = A(i, 0);
    int index2 = A(i, 1);

    // Get the labels from the respective indices
    int32_t label1 = labels1[index1];
    int32_t label2 = labels2[index2];
    
    // Verify semantic labels are consistent
    if (label1 == label2 && label1 != -1) {
      // Add the association and the label
      filtered_rows.emplace_back(Eigen::RowVector2i(index1, index2));
      filteredLabels.push_back(label1);
    }
  }

  // Convert the filtered rows to an Eigen matrix
  clipper::Association filteredA(filtered_rows.size(), 2);
  for (size_t i = 0; i < filtered_rows.size(); ++i) {
    filteredA.row(i) = filtered_rows[i];
  }

  return std::make_tuple(filteredA, filteredLabels);
}

// Filter based on set max # of associations
std::tuple<clipper::Association, std::vector<int> > downsampleAssociationMatrix(
  const clipper::Association& A,
  const std::vector<int> & corr_labels,
  const int max_associations) {
  int N = A.rows();
  int max_size_A = std::min(max_associations, N);  // avoid overflow

  // Generate random unique indices
  std::vector<int> indices(N);
  std::iota(indices.begin(), indices.end(), 0);  // fill with 0..N-1
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  
  // Downsample A
  std::vector<int> rand_ds_A_idxs(indices.begin(), indices.begin() + max_size_A);
  clipper::Association A_ds(max_size_A, 2);
  for (int i = 0; i < max_size_A; ++i) {
    A_ds.row(i) = A.row(rand_ds_A_idxs[i]);
  }
  
  // Downsample labels if provided
  std::vector<int>  corr_labels_ds;
  if (!corr_labels.empty()) {
    corr_labels_ds.resize(max_size_A);
    for (int i = 0; i < max_size_A; ++i) {
      corr_labels_ds[i] = corr_labels[rand_ds_A_idxs[i]];
    }
  }

  return std::make_tuple(A_ds, corr_labels_ds);
}

Eigen::Matrix4d umeyamaAlignment(const Eigen::MatrixXd& target, 
  const Eigen::MatrixXd& source) {
  assert(target.rows() == source.rows());
  
  // compute centroids
  Eigen::Vector3d target_mean = target.colwise().mean();
  Eigen::Vector3d source_mean = source.colwise().mean();
  
  // center points
  Eigen::MatrixXd target_centered = target.rowwise() - target_mean.transpose();
  Eigen::MatrixXd source_centered = source.rowwise() - source_mean.transpose();
  
  // compute covariance matrix
  Eigen::Matrix3d H = target_centered.transpose() * source_centered;
  
  // SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  
  // compute rotation
  Eigen::Matrix3d R = V * U.transpose();
  
  // ensure proper rotation (no reflection)
  if (R.determinant() < 0) {
    V.col(2) *= -1;
    R = V * U.transpose();
  }
  
  // compute translation
  Eigen::Vector3d t = source_mean - R * target_mean;
  
  // construct transformation matrix
  Eigen::Matrix4d Tfmat = Eigen::Matrix4d::Identity();
  Tfmat.block<3,3>(0,0) = R;
  Tfmat.block<3,1>(0,3) = t;

  return Tfmat;
}

Eigen::Affine3d computeTransformationFromInliers(
  const Eigen::MatrixXd& target_cloud, const Eigen::MatrixXd& source_cloud,
  const clipper::Association& corres) {
  int N = corres.rows();
  Eigen::MatrixXd target_corr(N, 3);
  Eigen::MatrixXd source_corr(N, 3);
  
  for (int i = 0; i < N; ++i) {
    int target_idx = corres(i, 0);
    int source_idx = corres(i, 1);

    target_corr.row(i) = target_cloud.row(target_idx);
    source_corr.row(i) = source_cloud.row(source_idx);
  }
  
  // align dem bad boys
  Eigen::Matrix4d tf_est = umeyamaAlignment(source_corr, target_corr);
  Eigen::Affine3d tf_est_affine(tf_est);
  
  return tf_est_affine;
}

class Lidar2OSM {
  public:
  /**
   * @brief Constructor that initializes Lidar2OSM with a configuration object.
   * @param config Configuration parameters for the matcher.
   */
  explicit Lidar2OSM(const Lidar2OSMConfig &config);

  /**
   * @brief Estimates the transformation between source and target point clouds.
   * @param src_cloud Source point cloud.
   * @param tgt_cloud Target point cloud.
   * @return The estimated registration solution.
   */
  Eigen::Affine3d estimate(Eigen::MatrixXd tgt_cloud, Eigen::MatrixXd src_cloud);
  
  private:
  // double min_dist_threshold_;
  // double eff_comms_dist_threshold_;
  // double map_voxel_size_;
  // int max_associations_;
  Lidar2OSMConfig config_;
};

}  // namespace lidar2osm