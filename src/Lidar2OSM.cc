// Lidar2OSM
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/KISSMatcher.hpp>

#include <pcl/kdtree/kdtree_flann.h>

#include "Lidar2OSM.hpp"
#include "utils.hpp"

lidar2osm::Lidar2OSM::Lidar2OSM(const Lidar2OSMConfig &config) {
  config_ = config;
//   reset();
}

Eigen::Affine3d lidar2osm::Lidar2OSM::estimate(Eigen::MatrixXd tgt_cloud, Eigen::MatrixXd src_cloud) {
  double eps = config_.map_voxel_size_ * 5;

  // instantiate the invariant function that will be used to score associations
  clipper::invariants::EuclideanDistance::Params iparams;
  iparams.epsilon = eps;
  iparams.sigma = 0.5 * iparams.epsilon;
  clipper::invariants::EuclideanDistancePtr invariant =
    std::make_shared<clipper::invariants::EuclideanDistance>(iparams);

  // set up CLIPPER rounding parameters
  clipper::Params params;
  params.rounding = clipper::Params::Rounding::DSD_HEU;
  
  // instantiate clipper object
  clipper::CLIPPER clipper(invariant, params);
  
  // create A2A associations
  // std::cout << "Filtering A: special filtering\n";
  // clipper::Association A_all_to_all = get_spec_a2a_assoc_matrix(target_cloud_orig, source_cloud_orig, 2.0);
  std::cout << "Creating A2A Associations \n";
  int target_points_len = tgt_cloud.rows();
  int source_points_len = src_cloud.rows();
//   clipper::Association A_all_to_all = lidar2osm::get_a2a_assoc_matrix(target_points_len, source_points_len);
  Eigen::Matrix<int, Eigen::Dynamic, 2> A_all_to_all(target_points_len, 2);
  int i = 0;
  for (int n1 = 0; n1 < target_points_len; ++n1) {
        A_all_to_all(i, 0) = n1;
        A_all_to_all(i, 1) = n1;
        ++i;
  }

  Eigen::MatrixXd tgt_cloud_points = tgt_cloud.leftCols(3);
  Eigen::MatrixXd src_cloud_points = src_cloud.leftCols(3);
  
//   // Filter associations based on ego dist of points
//   std::cout << "Filtering A: comms distance\n";
//   clipper::Association A_ego_filtered = lidar2osm::filter_by_ego_distance(
//     tgt_cloud_points, src_cloud_points, A_all_to_all, config_.eff_comms_dist_threshold_);

//   if (A_ego_filtered.rows() < 3) {
//     throw std::runtime_error(std::string("\033[1;31mToo many associations have been pruned. Remaining: ") 
//       + std::to_string(A_ego_filtered.rows()) + "\033[0m\n");
//   }

  // Filter associations based on semantic of points
  std::cout << "Filtering A: semantics\n";
  clipper::Association A_sem_filtered;
  std::vector<int> corr_filtered_labels;
  std::vector<int> target_labels(tgt_cloud.rows());
  
  for (int i = 0; i < tgt_cloud.rows(); ++i) {
    target_labels[i] = static_cast<int32_t>(tgt_cloud(i, 3));
  }
  std::vector<int> source_labels(src_cloud.rows());
  for (int i = 0; i < src_cloud.rows(); ++i) {
    source_labels[i] = static_cast<int32_t>(src_cloud(i, 3));
  }
  
//   std::tie(A_sem_filtered, corr_filtered_labels) = lidar2osm::filterSemanticAssociations(
//     target_labels, source_labels, A_ego_filtered);
  std::tie(A_sem_filtered, corr_filtered_labels) = lidar2osm::filterSemanticAssociations(
    target_labels, source_labels, A_all_to_all);
    
  // Filter associations based on a maximum # of associations
  std::cout << "Filtering A: max # specified associations\n";
  clipper::Association A_filtered;
  std::tie(A_filtered, corr_filtered_labels) = lidar2osm::downsampleAssociationMatrix(
    A_sem_filtered, corr_filtered_labels, config_.max_associations_);
//   std::tie(A_filtered, corr_filtered_labels) = lidar2osm::downsampleAssociationMatrix(
//     A_all_to_all, corr_filtered_labels, config_.max_associations_);

  // Score using invariant above and solve for maximal clique
  std::cout << "SCORING NOW\n";
  std::cout << "\033[1;31mNum Associations " << A_filtered.rows() << "\033[0m\n";

  clipper.scorePairwiseConsistency(tgt_cloud_points.transpose(), src_cloud_points.transpose(), A_filtered);
  // clipper::Affinity M = clipper.getAffinityMatrix();
  
  std::cout << "SOLVING NOW\n";
  // clipper.solve();
  clipper.solveAsMaximumClique();
  
  // Retrieve selected inliers
  clipper::Association Ainliers = clipper.getSelectedAssociations();
  std::cout << "Ainliers_len: " << Ainliers.rows() << "\n";
  
  // Compute peer2peer TF estimate
  std::cout << "COMPUTING TF\n";
  Eigen::Affine3d tf_est_affine = lidar2osm::computeTransformationFromInliers(tgt_cloud_points, src_cloud_points, Ainliers);
  
  return tf_est_affine;
}

// int main(int argc, char** argv) {
//   if (argc < 4) {
//     std::cerr << "Usage: " << argv[0] << " <src_pcd_file> <tgt_pcd_file> <resolution>" << std::endl;
//     return -1;
//   }

//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
//   pcl::PointCloud<pcl::PointXYZ>::Ptr src_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);

//   int num_iters = 10;
//   // for (int iter = 0; iter < num_iters; iter++) {
//     const std::string src_path = argv[1];
//     const std::string tgt_path = argv[2];
//     const float resolution     = std::stof(argv[3]);

//     if (!loadPointCloudRGB(src_path, src_pcl) || !loadPointCloudRGB(tgt_path, tgt_pcl)) {
//         std::cerr << "Error loading point cloud files." << std::endl;
//         return -1;
//     }
//     if (!loadPointCloud(src_path, src_pcl_xyz) || !loadPointCloud(tgt_path, tgt_pcl_xyz)) {
//         std::cerr << "Error loading point cloud files." << std::endl;
//         return -1;
//     }

//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_src_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_src_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);

//     Eigen::Matrix4f src_transform = getRandTF(0.0f, 10.0f, 0.0f, 360.0f);
//     pcl::transformPointCloud(*src_pcl, *rotated_src_pcl, src_transform);
//     src_pcl = rotated_src_pcl;

//     pcl::transformPointCloud(*src_pcl_xyz, *rotated_src_pcl_xyz, src_transform);
//     src_pcl_xyz = rotated_src_pcl_xyz;

//     kiss_matcher::KISSMatcherConfig km_config = kiss_matcher::KISSMatcherConfig(resolution);
//     km_config.robin_mode_ = "max_core"; // Options: "max_clique", "max_core", and "None"
//     kiss_matcher::KISSMatcher km_matcher(km_config);
    
//     lidar2osm::Lidar2OSMConfig config = lidar2osm::Lidar2OSMConfig(resolution);
//     lidar2osm::Lidar2OSM matcher(config);

//     // Get FPFH points from ROBIN
//     const auto& src_vec = convertCloudToVec(*src_pcl_xyz);
//     const auto& tgt_vec = convertCloudToVec(*tgt_pcl_xyz);
//     const auto km_matcher_solution = km_matcher.estimate(src_vec, tgt_vec);
//     Eigen::Affine3d km_matcher_est = Eigen::Affine3d::Identity();
//     km_matcher_est.linear()      = km_matcher_solution.rotation;     // set the 3×3 R
//     km_matcher_est.translation() = km_matcher_solution.translation;  // set the 3×1 t
//     const auto& src_vec_matched = km_matcher.src_matched_public_;
//     const auto& tgt_vec_matched = km_matcher.tgt_matched_public_;
//     size_t thres_num_inliers = 5;
//     if (km_matcher.getNumFinalInliers() < thres_num_inliers) {
//         std::cout << "\033[1;33m=> Registration might have failed :(\033[0m\n";
//     } else {
//         std::cout << "\033[1;32m=> Registration likely succeeded XD\033[0m\n";
//     }
//     km_matcher.print();
//     std::cout << "\n\n\033[1;32mLen src_vec_matched: " << src_vec_matched.size() << "\033[0m\n";

//     Eigen::MatrixXd tgt_vec_matched_eig = convertVecToEigen(tgt_vec_matched);
//     Eigen::MatrixXd src_vec_matched_eig = convertVecToEigen(src_vec_matched);

//     //   Eigen::MatrixXd tgt_cloud = convertCloudToEigen(tgt_pcl, 100000);
//     //   Eigen::MatrixXd src_cloud = convertCloudToEigen(src_pcl, 100000);

//     //     for (int i = 0; i < tgt_vec_matched_eig.rows(); ++i) {
//     //         for (int j = 0; j < tgt_cloud.rows(); ++j) {
//     //             double x = tgt_vec_matched_eig(i, 0);
//     //             double y = tgt_vec_matched_eig(i, 1);
//     //             double z = tgt_vec_matched_eig(i, 2);

//     //             double tgt_cloud_x = tgt_cloud(j, 0);
//     //             double tgt_cloud_y = tgt_cloud(j, 1);
//     //             double tgt_cloud_z = tgt_cloud(j, 2);
                
//     //             if (x == tgt_cloud_x && y == tgt_cloud_y && z == tgt_cloud_z) {
//     //                 // std::cout << "\033[1;32mMATCH!: \033[0m\n";
//     //                 tgt_vec_matched_eig(i, 3) = tgt_cloud(j, 3);
//     //                 break;
//     //             }
//     //         }
//     //     }

//     //     for (int i = 0; i < src_vec_matched_eig.rows(); ++i) {
//     //         for (int j = 0; j < src_cloud.rows(); ++j) {
//     //             double x = src_vec_matched_eig(i, 0);
//     //             double y = src_vec_matched_eig(i, 1);
//     //             double z = src_vec_matched_eig(i, 2);

//     //             double src_cloud_x = src_cloud(j, 0);
//     //             double src_cloud_y = src_cloud(j, 1);
//     //             double src_cloud_z = src_cloud(j, 2);
                
//     //             if (x == src_cloud_x && y == src_cloud_y && z == src_cloud_z) {
//     //                 // std::cout << "\033[1;32mMATCH!: \033[0m\n";
//     //                 src_vec_matched_eig(i, 3) = src_cloud(j, 3);
//     //                 break;
//     //             }
//     //         }
//     //     }

//     // Estimate rel tf using Lidar2OSM
//     Eigen::Affine3d ego_to_peer_est = matcher.estimate(tgt_vec_matched_eig, src_vec_matched_eig);
//     //   Eigen::Affine3d ego_to_peer_est = matcher.estimate(tgt_pcl, src_pcl);
//     Eigen::Affine3d src_transform_gt( src_transform.cast<double>() );
//     auto [km_matcher_RTE, km_matcher_RRE] = calcRRE(km_matcher_est, src_transform_gt);
//     auto [l2o_matcher_RTE, l2o_matcher_RRE] = calcRRE(ego_to_peer_est, src_transform_gt);

//     std::cout 
//     << km_matcher_RTE << " " << km_matcher_RRE << "\n"
//     << l2o_matcher_RTE << " " << l2o_matcher_RRE << "\n";

//     // std::cout 
//     // << "KM Translational error: " << km_matcher_RTE << " m\n"
//     // << "L2O Translational error: " << l2o_matcher_RTE << " m\n"
//     // << "KM Rotational error: "    << km_matcher_RRE << "°)\n"
//     // << "L2O Rotational error: "    << l2o_matcher_RRE << "°)\n";

//         // if (km_matcher_solution.valid) {
//         //     std::cout << "\033[1;32m=> Registration likely succeeded XD\033[0m\n";
//         // }
//   // }

//   Eigen::Matrix4f solution_eigen = ego_to_peer_est.matrix().cast<float>();
//   pcl::PointCloud<pcl::PointXYZ>::Ptr est_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//   pcl::transformPointCloud(*src_pcl_xyz, *est_cloud, solution_eigen);
//   std::filesystem::path src_file_path(src_path);
//   std::string warped_pcd_filename =
//       src_file_path.parent_path().string() + "/" + src_file_path.stem().string() + "_warped.pcd";
//   pcl::io::savePCDFileASCII(warped_pcd_filename, *est_cloud);
//   std::cout << "Saved transformed source point cloud to: " << warped_pcd_filename << std::endl;

//   // Visualize global peer cloud transformed
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr peer_orig_global_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  
//   int max_points_global_cloud_ = 100000;
//   Eigen::MatrixXd source_global_cloud_orig = convertCloudToEigen(src_pcl, max_points_global_cloud_);
// //   Eigen::Affine3d tf_to_world =  map1_to_ego * ego_to_peer_est * peer_to_map2;
//   Eigen::MatrixXd source_cloud_est = transformEigenCloud(source_global_cloud_orig, ego_to_peer_est);
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_source_cloud_est = convertEigenToCloud(source_cloud_est);

//   colorize(*tgt_pcl, {0, 255, 0});
//   colorize(*pcl_source_cloud_est, {255, 0, 0});

//   pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
//   viewer1.addPointCloud<pcl::PointXYZRGB>(pcl_source_cloud_est, "src_white");
//   viewer1.addPointCloud<pcl::PointXYZRGB>(tgt_pcl, "tgt_colored");

//   while (!viewer1.wasStopped()) {
//     viewer1.spin();
//   }

//   return 0;
// }

void saveResults(double km_matcher_RTE, double km_matcher_RRE, 
  double l2o_matcher_RTE, double l2o_matcher_RRE, std::string csvFilepath) {
  // Write results to CSV without overwriting previous lines
  std::ofstream results_csv;
  results_csv.open(csvFilepath, std::ios::app); // Open in append mode

  // Check if the file opened correctly
  if (!results_csv.is_open()) {
      std::cerr << "Failed to open results.csv file for writing." << std::endl;
  }

  // If the file is empty (first line), you might want to write headers
  // This optional check ensures the header is only written if the file is empty.
  if (results_csv.tellp() == 0) {
      results_csv << "km_matcher_RTE,km_matcher_RRE,l2o_matcher_RTE,l2o_matcher_RRE\n";
  }

  // Write the values you computed
  results_csv << km_matcher_RTE << "," 
              << km_matcher_RRE << "," 
              << l2o_matcher_RTE << "," 
              << l2o_matcher_RRE << "\n";

  // Close the CSV file stream
  results_csv.close();

  std::cout << "Results appended to results.csv" << std::endl;
}


int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <src_pcd_file> <tgt_pcd_file> <csv_results_file> <resolution>" << std::endl;
    return -1;
  }

  const std::string src_path = argv[1];
  const std::string tgt_path = argv[2];
  const std::string resultsPath = argv[3];
  const float resolution     = std::stof(argv[4]);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);

  if (!loadPointCloudRGB(src_path, src_pcl) || !loadPointCloudRGB(tgt_path, tgt_pcl)) {
      std::cerr << "Error loading point cloud files." << std::endl;
      return -1;
  }
  if (!loadPointCloud(src_path, src_pcl_xyz) || !loadPointCloud(tgt_path, tgt_pcl_xyz)) {
      std::cerr << "Error loading point cloud files." << std::endl;
      return -1;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_src_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_src_pcl_xyz(new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Matrix4f src_transform = getRandTF(0.0f, 10.0f, 0.0f, 360.0f);
  pcl::transformPointCloud(*src_pcl, *rotated_src_pcl, src_transform);
  src_pcl = rotated_src_pcl;

  pcl::transformPointCloud(*src_pcl_xyz, *rotated_src_pcl_xyz, src_transform);
  src_pcl_xyz = rotated_src_pcl_xyz;

  kiss_matcher::KISSMatcherConfig km_config = kiss_matcher::KISSMatcherConfig(resolution);
  km_config.robin_mode_ = "max_core"; // Options: "max_clique", "max_core", and "None"
  kiss_matcher::KISSMatcher km_matcher(km_config);
  
  lidar2osm::Lidar2OSMConfig config = lidar2osm::Lidar2OSMConfig(resolution);
  lidar2osm::Lidar2OSM matcher(config);

  // Get FPFH points from ROBIN
  const auto& src_vec = convertCloudToVec(*src_pcl_xyz);
  const auto& tgt_vec = convertCloudToVec(*tgt_pcl_xyz);
  const auto km_matcher_solution = km_matcher.estimate(src_vec, tgt_vec);
  Eigen::Affine3d km_matcher_est = Eigen::Affine3d::Identity();
  km_matcher_est.linear()      = km_matcher_solution.rotation;     // set the 3×3 R
  km_matcher_est.translation() = km_matcher_solution.translation;  // set the 3×1 t
  const auto& src_vec_matched = km_matcher.src_matched_public_;
  const auto& tgt_vec_matched = km_matcher.tgt_matched_public_;
  size_t thres_num_inliers = 5;
  if (km_matcher.getNumFinalInliers() < thres_num_inliers) {
      std::cout << "\033[1;33m=> Registration might have failed :(\033[0m\n";
  } else {
      std::cout << "\033[1;32m=> Registration likely succeeded XD\033[0m\n";
  }
  km_matcher.print();
  std::cout << "\n\n\033[1;32mLen src_vec_matched: " << src_vec_matched.size() << "\033[0m\n";

  Eigen::MatrixXd tgt_vec_matched_eig = convertVecToEigen(tgt_vec_matched);
  Eigen::MatrixXd src_vec_matched_eig = convertVecToEigen(src_vec_matched);

  // Estimate rel tf using Lidar2OSM
  Eigen::Affine3d ego_to_peer_est = matcher.estimate(tgt_vec_matched_eig, src_vec_matched_eig);
  //   Eigen::Affine3d ego_to_peer_est = matcher.estimate(tgt_pcl, src_pcl);
  Eigen::Affine3d src_transform_gt( src_transform.cast<double>() );
  auto [km_matcher_RTE, km_matcher_RRE] = calcRRE(km_matcher_est, src_transform_gt);
  auto [l2o_matcher_RTE, l2o_matcher_RRE] = calcRRE(ego_to_peer_est, src_transform_gt);

  std::cout 
  << km_matcher_RTE << " " << km_matcher_RRE << "\n"
  << l2o_matcher_RTE << " " << l2o_matcher_RRE << "\n";

  // Write results to CSV without overwriting previous lines
  saveResults(km_matcher_RTE, km_matcher_RRE, l2o_matcher_RTE, l2o_matcher_RRE, resultsPath);

  // // Save estimated pcd
  // Eigen::Matrix4f solution_eigen = ego_to_peer_est.matrix().cast<float>();
  // pcl::PointCloud<pcl::PointXYZ>::Ptr est_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::transformPointCloud(*src_pcl_xyz, *est_cloud, solution_eigen);
  // std::filesystem::path src_file_path(src_path);
  // std::string warped_pcd_filename =
  //     src_file_path.parent_path().string() + "/" + src_file_path.stem().string() + "_warped.pcd";
  // pcl::io::savePCDFileASCII(warped_pcd_filename, *est_cloud);
  // std::cout << "Saved transformed source point cloud to: " << warped_pcd_filename << std::endl;

  // // Visualize global peer cloud transformed
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr peer_orig_global_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  
  // int max_points_global_cloud_ = 100000;
  // Eigen::MatrixXd source_global_cloud_orig = convertCloudToEigen(src_pcl, max_points_global_cloud_);
  // //   Eigen::Affine3d tf_to_world =  map1_to_ego * ego_to_peer_est * peer_to_map2;
  // Eigen::MatrixXd source_cloud_est = transformEigenCloud(source_global_cloud_orig, ego_to_peer_est);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_source_cloud_est = convertEigenToCloud(source_cloud_est);

  // colorize(*tgt_pcl, {0, 255, 0});
  // colorize(*pcl_source_cloud_est, {255, 0, 0});

  // pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
  // viewer1.addPointCloud<pcl::PointXYZRGB>(pcl_source_cloud_est, "src_white");
  // viewer1.addPointCloud<pcl::PointXYZRGB>(tgt_pcl, "tgt_colored");

  // while (!viewer1.wasStopped()) {
  //   viewer1.spin();
  // }

  return 0;
}