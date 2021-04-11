import argparse
import copy

import numpy as np
import open3d as o3d

LOG = True


def draw_registration_result(source, target, transformation, window_name="Result"):
    """
    Displays registration result
    :param window_name: name of window
    :param source: source PointCloud
    :param target: target PointCloud
    :param transformation: transformation from target to source
    :return:
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)


def preprocess_point_cloud(pcd, voxel_size):
    """
    Resamples point cloud and computes normals
    :param pcd: point cloud
    :param voxel_size: size of voxel
    :return: resampled pcd and features
    """
    if LOG:
        print("INFO: Downsampling with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    if LOG:
        print("INFO: Estimating normals with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    if LOG:
        print("INFO: Computing FPFH features with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, file1, file2):
    """
    Loads and prepares dataset
    :param voxel_size: size of voxel to resample
    :param file1:
    :param file2:
    :return: pcds, resampled pcds, features
    """
    if LOG:
        print("INFO: Load two point clouds.")
    # "./data/data10_points.ply"
    # "./data/headFace3_geo_low.ply"
    source = o3d.io.read_point_cloud(file1)
    target = o3d.io.read_point_cloud(file2)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Excecutes global registration using RANSAC
    :param source_down: resampled pcd
    :param target_down: resampled pcd
    :param source_fpfh: features
    :param target_fpfh: features
    :param voxel_size: size of voxel
    :return:
    """
    distance_threshold = voxel_size * 0.5
    if LOG:
        print("INFO: Launching global registration using RANSAC")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    if LOG:
        print("INFO: Apply fast global registration with distance threshold %.3f" \
              % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,
                        result_ransac, distance_threshold=None):
    if not distance_threshold:
        distance_threshold = voxel_size * 0.3
    if LOG:
        print("INFO: Running point-to-plane ICP registration")
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process global registration of point clouds and saves '
                                                 'result transformation matrix to "finalreg.txt"')
    parser.add_argument('source', metavar='source', help='source point cloud')
    parser.add_argument('target', metavar='target', help='target point cloud')
    parser.add_argument('--voxel', '-v', dest='voxel', action='store',
                        help='voxel size for image processing. If result is not good, you can try to increase it. '
                             'Default value is 5')
    parser.add_argument('--silent', '-s', dest='silent', action='store_false',
                        help='if is set, you will not see debugging message')
    parser.add_argument('--no_preview', '-np', dest='preview', action='store_false',
                        help='if is set, you will preview registration result')

    args = parser.parse_args()
    if args.voxel is None:
        voxel_size = 5
    else:
        voxel_size = args.voxel
    LOG = args.silent
    # file1 = "./data/data10_points.ply"
    # file2 = "./data/headFace3_geo_low.ply"
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,
                                                                                         args.source, args.target)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    if args.preview:
        draw_registration_result(source_down, target_down, result_ransac.transformation, window_name="Pre-registration")
    print(result_ransac)
    np.savetxt("prereg.txt", result_ransac.transformation, fmt="%d,")
    # result_fast = execute_fast_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    # print(result_fast.transformation)
    # draw_registration_result(source_down, target_down, result_fast.transformation)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_ransac, 0.002)
    np.savetxt("finalreg.txt", result_icp.transformation, fmt="%20f,")
    print(result_icp)
    if args.preview:
        draw_registration_result(source, target, result_icp.transformation, window_name="Registration Result")
