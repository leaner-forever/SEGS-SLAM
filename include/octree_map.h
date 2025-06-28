#pragma once

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree.h>
#include <pcl/types.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class CustomOctree : public pcl::octree::OctreePointCloud<PointT> {
public:
    CustomOctree(double resolution) : pcl::octree::OctreePointCloud<PointT>(resolution) {}

    bool addPointToCloud(PointCloudT::Ptr& cloud_in, PointCloudT::Ptr &cloud_full);
};