import ast
import json
import math
import rtree
import networkx as nx
import numpy as np
import osmnx as ox
from os.path import join
import logging
import pyproj
import rasterio
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from shapely import wkt
from shapely.geometry import LineString, Polygon, Point, box
from scipy.spatial import Voronoi
from geopy.distance import geodesic
from typing import Dict
from tqdm import tqdm
from shapely.ops import unary_union
from argparse import ArgumentParser


def grid_partition(data_feature: Dict, cell_len=2000):
    min_lon, min_lat = data_feature['min_lon'], data_feature['min_lat']
    max_lon, max_lat = data_feature['max_lon'], data_feature['max_lat']
    utm_epsg = data_feature['utm_epsg']
    latlon2xy = pyproj.Transformer.from_crs(4326, utm_epsg)
    min_x, min_y = latlon2xy.transform(min_lat, min_lon)
    max_x, max_y = latlon2xy.transform(max_lat, max_lon)
    x_num, y_num = math.ceil((max_x - min_x) / cell_len), math.ceil((max_y - min_y) / cell_len)
    d_lon, d_lat = (max_lon - min_lon) / x_num, (max_lat - min_lat) / y_num
    polygons = []
    for x in range(x_num):
        for y in range(y_num):
            polygon = Polygon([
                (min_lon + (x + cx) * d_lon, min_lat + (y + cy) * d_lat)
                for cx, cy in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
            ])
            polygons.append(polygon)
    grid_gdf = gpd.GeoDataFrame(geometry=polygons)
    grid_gdf['region_id'] = range(len(grid_gdf))
    return grid_gdf[['region_id', 'geometry']]


def get_region_rel(region_df: pd.DataFrame):
    """
    计算 region 之间的邻接关系和距离
    """
    # TODO: 检查邻接关系判断逻辑
    def check_adj(polygon1: Polygon, polygon2: Polygon):
        return polygon1.intersects(polygon2)

    def calc_dist(polygon1: Polygon, polygon2: Polygon):
        lon1, lat1 = polygon1.centroid.x, polygon1.centroid.y
        lon2, lat2 = polygon2.centroid.x, polygon2.centroid.y
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    rel_list = []
    for _, A in region_df.iterrows():
        for _, B in region_df.iterrows():
            if B['region_id'] == A['region_id']:
                continue
            is_adj = check_adj(A['geometry'], B['geometry'])
            dist = calc_dist(A['geometry'], B['geometry'])
            rel_list.append((A['region_id'], B['region_id'], int(is_adj), dist))
    # Note: 返回两两之间的距离以及是否邻接
    adj_df = pd.DataFrame(data=rel_list, columns=['ori', 'des', 'is_adj', 'dist'])
    return adj_df


def get_pop_data(pop_raster, min_lon, min_lat, max_lon, max_lat):
    """
    读取一个城市范围内的人口数据
    """
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:4326")
    geo = geo.to_crs(pop_raster.crs)

    image, transform = mask(pop_raster, geo.geometry, crop=True)
    image = image[0]  # 一个 channel
    height, width = image.shape[0], image.shape[1]
    data_list = []
    for i in range(height):
        for j in range(width):
            lon, lat = rasterio.transform.xy(transform, i, j)
            data_list.append((lon, lat, image[i, j]))
    pop_df = pd.DataFrame(data=data_list, columns=['lon', 'lat', 'population'])
    pop_df['population'] = pop_df['population'].apply(lambda x: x if x > 0 else 0)
    return pop_df


def pop_to_region(pop_df: pd.DataFrame, region_df: pd.DataFrame):
    """
    计算每个 region 的人口数量和人口密度
    :param pop_df: "lon", "lat", "population", "area"
    :param region_df: "geometry" (Shape)
    :return:
    """
    region_df['population'] = 0
    for _, pop_row in tqdm(pop_df.iterrows(), total=pop_df.shape[0], desc='pop to region'):
        point = Point(pop_row['lon'], pop_row['lat'])
        for idx, region_row in region_df.iterrows():
            if region_row['geometry'].contains(point):
                region_df.loc[idx, 'population'] += pop_row['population']
                break
    region_df['population_density'] = region_df['population'] / (region_df['area'] + 1e-8)
    return region_df


def poi_to_region(poi_df: pd.DataFrame, region_df: pd.DataFrame, poi_type_num: int):
    """
    :param poi_df: "poi_id", "geometry" (Shape), "poi_type" (int)
    :param region_df: "region_id", "geometry" (Shape)
    :param poi_type_num:
    :return:
    """
    poi_df = poi_df.set_index('poi_id', drop=False)
    region_df = region_df.set_index('region_id', drop=False)

    # 构建 poi Rtree
    poi_rtree = rtree.index.Index()
    for poi_id, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0], desc='poi rtree'):
        poi_rtree.insert(id=int(poi_id), coordinates=row['geometry'].bounds, obj=row['geometry'])

    # 查询与每个 region 相交的 poi
    for i in range(poi_type_num):
        region_df[f'poi{i}'] = 0
    for region_id, row in region_df.iterrows():
        candidates = list(poi_rtree.intersection(row['geometry'].bounds))
        attached_ids = [poi_id for poi_id in candidates if poi_df.loc[poi_id, 'geometry'].intersects(row['geometry'])]
        for poi_id in attached_ids:
            poi_type = poi_df.loc[poi_id, 'poi_type']
            region_df.loc[region_id, f'poi{poi_type}'] += 1
    return region_df


def road_to_region(road_df: pd.DataFrame, region_df: pd.DataFrame, road_type_num: int):
    """
    计算每个 region 的路网特征
    :param road_df: "road_id", "geometry" (Shape)
    :param region_df: "region_id", "geometry" (Shape)
    :return:
    """
    road_df = road_df.set_index('road_id', drop=False)
    region_df = region_df.set_index('region_id', drop=False)

    # 构建道路 Rtree
    road_rtree = rtree.index.Index()
    for road_id, row in road_df.iterrows():
        road_rtree.insert(id=int(road_id), coordinates=row['geometry'].bounds, obj=row['geometry'])

    # 道路匹配到 region
    region_to_road = dict()
    for region_id, row in region_df.iterrows():
        candidates = list(road_rtree.intersection(row['geometry'].bounds))
        region_to_road[region_id] = [
            road_id for road_id in candidates if road_df.loc[road_id, 'geometry'].intersects(row['geometry'])
        ]

    # 聚合特征
    for road_type in range(road_type_num):
        region_df[f'road_num_{road_type}'] = 0
        region_df[f'road_length_{road_type}'] = 0
    region_df['road_num'] = 0
    region_df['road_length'] = 0
    # degree 指的是 degree centrality
    region_df['mean_in_degree'] = 0
    region_df['mean_out_degree'] = 0
    region_df['mean_degree'] = 0
    for region_id, road_list in region_to_road.items():
        sub_road_df = road_df.loc[road_list, :]
        region_df.loc[region_id, 'road_num'] = len(sub_road_df)
        region_df.loc[region_id, 'road_length'] = sub_road_df['length'].sum()
        region_df.loc[region_id, 'mean_in_degree'] = sub_road_df['in_degree'].mean()
        region_df.loc[region_id, 'mean_out_degree'] = sub_road_df['out_degree'].mean()
        region_df.loc[region_id, 'mean_degree'] = sub_road_df['degree'].mean()
        for road_type, group in sub_road_df.groupby('highway'):
            region_df.loc[region_id, f'road_num_{road_type}'] = len(group)
            region_df.loc[region_id, f'road_length_{road_type}'] = group['length'].sum()

    return region_df


def region_mask(region_df):
    """
    栅格切分的 region 可能位于水域, 影响整体表征的对齐
    特征抽取完毕后再 mask
    """
    feature_cols = (['population']
                    + [f'road_num_{i}' for i in range(14)]
                    + [f'road_length_{i}' for i in range(14)])
    # note: 返回 mask 列的类型为 bool
    region_df['mask'] = region_df.apply(lambda row: np.count_nonzero(row[feature_cols]) == 0, axis=1)
    return region_df


def prepare_region_feature(city, cell_len=2000):
    print(f'Extract region feature in {city} ...... ')
    data_dir = 'cleared_data'
    data_feature = json.load(open(join(data_dir, city, 'data_feature.json'), 'r'))

    region_df = grid_partition(data_feature, cell_len=cell_len)

    poi_df = pd.read_csv(join(data_dir, city, 'poi.csv'))
    poi_df['geometry'] = poi_df['geometry'].apply(wkt.loads)
    road_df = pd.read_csv(join(data_dir, city, 'road_feature.csv'))  # road feature
    road_df['geometry'] = road_df['geometry'].apply(wkt.loads)

    # 人口数据
    if city in ['bj', 'xa', 'cd', 'wh', 'sz']:
        pop_file = 'china_population.tif'
    elif city in ['ny', 'dc', 'chi', 'sf']:
        pop_file = 'usa_ppp_2020_UNadj_constrained.tif'
    elif city in ['toronto']:
        pop_file = 'can_ppp_2020_UNadj_constrained.tif'
    else:
        print('Country error ...... ')
        return
    pop_raster = rasterio.open(join('ori_data', pop_file))
    pop_df = get_pop_data(
        pop_raster, data_feature['min_lon'], data_feature['min_lat'], data_feature['max_lon'], data_feature['max_lat']
    )

    # 空间特征
    region_df['area'] = (gpd.GeoSeries(region_df['geometry'], crs=4326)
                         .to_crs(data_feature['utm_epsg']).apply(lambda polygon: polygon.area))
    region_df['lon'] = region_df['geometry'].apply(lambda polygon: polygon.centroid.x)
    region_df['lat'] = region_df['geometry'].apply(lambda polygon: polygon.centroid.y)
    c_lon = (data_feature['min_lon'] + data_feature['max_lon']) / 2
    c_lat = (data_feature['min_lat'] + data_feature['max_lat']) / 2
    region_df['dist_to_center'] = region_df.apply(
        lambda row: geodesic((row['lat'], row['lon']), (c_lat, c_lon)).meters, axis=1
    )

    # 人口特征
    region_df = pop_to_region(pop_df, region_df)

    # poi 特征
    region_df = poi_to_region(poi_df, region_df, poi_type_num=15)  # 注意 POI 类型数量

    # 路网特征
    region_df = road_to_region(road_df, region_df, road_type_num=14)  # 注意 road 类型数量

    # 排除 无人/水域 的栅格
    region_df = region_mask(region_df)
    region_df = region_df[~region_df['mask']].copy()
    del region_df['mask']
    region_df['region_id'] = range(len(region_df))

    # 计算两两之间的距离和邻接关系
    rel_df = get_region_rel(region_df)

    region_df[['region_id', 'geometry']].to_csv(join(data_dir, city, 'grid_region.csv'), index=False)
    rel_df.to_csv(join(data_dir, city, 'grid_region_rel.csv'), index=False)
    region_df.to_csv(join(data_dir, city, 'grid_region_feature.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--city', type=str)
    args = parser.parse_args()

    prepare_region_feature(args.city, cell_len=2000)
