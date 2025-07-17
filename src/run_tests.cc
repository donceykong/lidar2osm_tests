#include <filesystem>

// PCL
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
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

// Internal
#include "utils.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <src_pcd_file> <tgt_pcd_file> <resolution> <yaw_aug_angle>" << std::endl;
    return -1;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_pcl(new pcl::PointCloud<pcl::PointXYZ>);

  const std::string src_path = argv[1];
  const std::string tgt_path = argv[2];
  const float resolution     = std::stof(argv[3]);

  Eigen::Matrix4f yaw_transform = Eigen::Matrix4f::Identity();
  if (argc > 4) {

    float yaw_aug_angle = std::stof(argv[4]);             // Yaw angle in degrees
    float yaw_rad       = yaw_aug_angle * M_PI / 180.0f;  // Convert to radians

    std::cout << "YAW ANGLE IS: " << yaw_aug_angle << " Degrees.\n";

    yaw_transform(0, 0) = std::cos(yaw_rad);
    yaw_transform(0, 1) = -std::sin(yaw_rad);
    yaw_transform(1, 0) = std::sin(yaw_rad);
    yaw_transform(1, 1) = std::cos(yaw_rad);
  }
  // ——— set up RNG and distributions ———
  std::random_device rd;
  std::mt19937 gen(rd());

  // translation in [0,100]
  std::uniform_real_distribution<float> dist_trans(10.0f, 100.0f);

  // rotation angle in [45°,180°] (in radians)
  // std::uniform_real_distribution<float> dist_angle(100.0f * M_PI / 180.0f, 270.0f * M_PI / 180.0f);
  std::uniform_real_distribution<float> dist_angle(180.0f * M_PI / 180.0f, 180.0f * M_PI / 180.0f);

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

  yaw_transform = X;
  std::cout << "Random transform X:\n" << X << "\n";
  std::cout << "Random transform yaw_transform:\n" << yaw_transform << "\n";

  std::cout << "Source input: " << src_path << "\n";
  std::cout << "Target input: " << tgt_path << "\n";

  if (!loadPointCloud(src_path, src_pcl) || !loadPointCloud(tgt_path, tgt_pcl)) {
    std::cerr << "Error loading point cloud files." << std::endl;
    return -1;
  }

  std::vector<int> src_indices;
  std::vector<int> tgt_indices;
  pcl::removeNaNFromPointCloud(*src_pcl, *src_pcl, src_indices);
  pcl::removeNaNFromPointCloud(*tgt_pcl, *tgt_pcl, tgt_indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_src_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*src_pcl, *rotated_src_pcl, yaw_transform);
  src_pcl = rotated_src_pcl;

  const auto& src_vec = convertCloudToVec(*src_pcl);
  const auto& tgt_vec = convertCloudToVec(*tgt_pcl);

  std::cout << "Number of elements src_vec: " << src_vec.size() << "\n";   // prints 5
  std::cout << "Number of elements tgt_vec: " << tgt_vec.size() << "\n";   // prints 5
  std::cout << "\033[1;32mLoad complete!\033[0m\n";

  kiss_matcher::KISSMatcherConfig config = kiss_matcher::KISSMatcherConfig(resolution);
  // NOTE(hlim): Two important parameters for enhancing performance
  // 1. `config.use_quatro_ (default: false)`
  // If the rotation is predominantly around the yaw axis, set `use_quatro_` to true like:
  // config.use_quatro_ = true;
  // Otherwise, the default mode activates SO(3)-based GNC.
  // E.g., in the case of `VBR-Collosseo`, it should be set as `false`.
  //
  // 2. `config.use_ratio_test_ (default: true)`
  // If dealing with a scan-level point cloud, the impact of `use_ratio_test_` is insignificant.
  // Plus, setting `use_ratio_test_` to false helps speed up slightly
  // If you want to try your own scan at a scan-level or loop closing situation,
  // setting `false` boosts the inference speed.
  // config.use_ratio_test_ = false;
  config.robin_mode_ = "max_core"; // Options: "max_clique", "max_core", and "None"
  kiss_matcher::KISSMatcher matcher(config);

  size_t thres_num_inliers = 5;
  size_t num_final_inliers = 0;
  pcl::PointCloud<pcl::PointXYZ> src_viz = *src_pcl;
  pcl::PointCloud<pcl::PointXYZ> tgt_viz = *tgt_pcl;
  pcl::PointCloud<pcl::PointXYZ> est_viz;
  Eigen::Matrix4f solution_eigen = Eigen::Matrix4f::Identity();
  while (num_final_inliers < thres_num_inliers) {
    // ——— sample translation ———
    Eigen::Vector3f t;
    t << dist_trans(gen),
         dist_trans(gen),
         dist_trans(gen);
    
    std::cout << "\n\nt: " << t << "\n";

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

    yaw_transform = X;
    pcl::transformPointCloud(*src_pcl, *rotated_src_pcl, yaw_transform);
    src_pcl = rotated_src_pcl;

    const auto& src_vec = convertCloudToVec(*src_pcl);
    src_viz = *src_pcl;

    const auto solution = matcher.estimate(src_vec, tgt_vec);

    // Visualization
    // pcl::PointCloud<pcl::PointXYZ> src_viz = *src_pcl;
    // pcl::PointCloud<pcl::PointXYZ> tgt_viz = *tgt_pcl;
    // pcl::PointCloud<pcl::PointXYZ> est_viz;

    // Eigen::Matrix4f solution_eigen      = Eigen::Matrix4f::Identity();
    solution_eigen.block<3, 3>(0, 0)    = solution.rotation.cast<float>();
    solution_eigen.topRightCorner(3, 1) = solution.translation.cast<float>();

    matcher.print();

    size_t num_rot_inliers   = matcher.getNumRotationInliers();
    // size_t num_final_inliers = matcher.getNumFinalInliers();
    num_final_inliers = matcher.getNumFinalInliers();
  }

  // NOTE(hlim): By checking the final inliers, we can determine whether
  // the registration was successful or not. The larger the threshold,
  // the more conservatively the decision is made.
  // See https://github.com/MIT-SPARK/KISS-Matcher/issues/24
  // size_t thres_num_inliers = 5;
  if (num_final_inliers < thres_num_inliers) {
    std::cout << "\033[1;33m=> Registration might have failed :(\033[0m\n";
  } else {
    std::cout << "\033[1;32m=> Registration likely succeeded XD\033[0m\n";
  }

  std::cout << solution_eigen << std::endl;
  std::cout << "=====================================" << std::endl;

  // ------------------------------------------------------------
  // Save warped source cloud
  // ------------------------------------------------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr est_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*src_pcl, *est_cloud, solution_eigen);
  std::filesystem::path src_file_path(src_path);
  std::string warped_pcd_filename =
      src_file_path.parent_path().string() + "/" + src_file_path.stem().string() + "_warped.pcd";
  pcl::io::savePCDFileASCII(warped_pcd_filename, *est_cloud);
  std::cout << "Saved transformed source point cloud to: " << warped_pcd_filename << std::endl;

  // ------------------------------------------------------------
  //
  // ------------------------------------------------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_keypoints_viz = convertVecToCloud(matcher.tgt_matched_public_);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_keypoints_viz_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  colorize(*tgt_keypoints_viz, *tgt_keypoints_viz_colored, {0, 255, 0});

  pcl::PointCloud<pcl::PointXYZ>::Ptr src_keypoints_viz = convertVecToCloud(matcher.src_matched_public_);

  pcl::transformPointCloud(*src_keypoints_viz, *src_keypoints_viz, solution_eigen);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_keypoints_viz_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  colorize(*src_keypoints_viz, *src_keypoints_viz_colored, {255, 0, 0});

  pcl::visualization::PCLVisualizer viewerNEW("Simple Cloud Viewer");
  viewerNEW.addPointCloud<pcl::PointXYZRGB>(tgt_keypoints_viz_colored, "tgt_green");
  viewerNEW.addPointCloud<pcl::PointXYZRGB>(src_keypoints_viz_colored, "src_red");

  // -- draw one line per correspondence --
  for (std::size_t i = 0; i < src_keypoints_viz_colored->points.size(); ++i)
  {
      // grab endpoints
      const pcl::PointXYZRGB& p_src = src_keypoints_viz_colored->points[i];
      const pcl::PointXYZRGB& p_tgt = tgt_keypoints_viz_colored->points[i];

      // make a unique id for each line
      std::string line_id = "corr_line_" + std::to_string(i);

      // add a white line from src→tgt
      viewerNEW.addLine<pcl::PointXYZRGB>(
        p_src, p_tgt,     // endpoints
        1.0, 1.0, 1.0,    // RGB color (white)
        line_id           // unique identifier
      );

      // (optional) make it thicker so you can see it better
      viewerNEW.setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
        2,                // width in pixels
        line_id
      );
  }

  while (!viewerNEW.wasStopped()) {
    viewerNEW.spin();
  }

  // ------------------------------------------------------------
  // Visualization
  // ------------------------------------------------------------
  pcl::transformPointCloud(src_viz, est_viz, solution_eigen);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr est_q_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

  colorize(src_viz, *src_colored, {195, 195, 195});
  colorize(tgt_viz, *tgt_colored, {89, 167, 230});
  colorize(est_viz, *est_q_colored, {238, 160, 61});

  pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
  viewer1.addPointCloud<pcl::PointXYZRGB>(src_colored, "src_red");
  viewer1.addPointCloud<pcl::PointXYZRGB>(tgt_colored, "tgt_green");
  viewer1.addPointCloud<pcl::PointXYZRGB>(est_q_colored, "est_q_blue");
  
  while (!viewer1.wasStopped()) {
    viewer1.spin();
  }

  return 0;
}
