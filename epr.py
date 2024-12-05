import random

import pandas as pd
import numpy as np
import math
import geopandas as gpd
import skmob
import yaml
from skmob.models.epr import DensityEPR
from typing import List
from os.path import join
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


class ActEPR:
    def __init__(self, cfg, aoi_gdf: gpd.GeoDataFrame, act_time_distri: np.ndarray):
        self.rho = cfg['rho']
        self.gamma = cfg['gamma']
        self.total_loc_num = len(aoi_gdf)
        self.act_time_distri = act_time_distri
        self.aoi_gdf = aoi_gdf.to_crs(32649)
        self.aoi_gdf['x'] = self.aoi_gdf['geometry'].apply(lambda geom: geom.centroid.x)
        self.aoi_gdf['y'] = self.aoi_gdf['geometry'].apply(lambda geom: geom.centroid.y)
        dist_mx = pdist(X=self.aoi_gdf[['x', 'y']].values, metric='euclidean')
        self.dist_mx = squareform(dist_mx)
        self.total_locs = set(self.aoi_gdf['aoi_id'])

    def _get_candidates(self, visited_locs, act):
        possible_locs = self.total_locs - set(visited_locs)
        candidate_df = self.aoi_gdf.iloc[possible_locs]
        candidate_df = candidate_df[candidate_df[f'act{act}'] > 0]
        return candidate_df

    def _time_sample(self, act, prev_time):
        dur_distri = self.act_time_distri[act]
        dur = random.sample(dur_distri, 1)[0]
        cur_time = prev_time + dur
        return cur_time

    def _choose_start(self, act):
        aoi_df = self._get_candidates(visited_locs=[], act=act)
        start_aoi_id = random.choice(aoi_df['aoi_id'])
        start_time = self._time_sample(act, 0)
        return {
            'time': start_time,
            'act': act,
            'aoi_id': start_aoi_id
        }

    def _explore(self, cur_act, current_traj):
        cur_idx = len(current_traj)

        # 时间采样
        prev_time = current_traj[-1]['time']
        cur_time = self._time_sample(cur_act, prev_time)

        # 候选 AOI 获取
        visited_locs = [rcd['aoi_id'] for rcd in current_traj]
        aoi_df = self._get_candidates(visited_locs, cur_act)

        if len(aoi_df) == 0:
            return None

        # 空间距离计算
        prev_aoi_id = current_traj[-1]['aoi_id']
        aoi_df['dist'] = aoi_df['aoi_id'].apply(lambda aoi_id: self.dist_mx[prev_aoi_id, aoi_id])

        # 根据出行吸引力采样
        aoi_df['population'] = aoi_df[f'act{cur_act}']
        prob = (aoi_df['population'] / (aoi_df['dist'] + 1e-8)).to_numpy()
        prob = prob / prob.sum()
        sampled_idx = np.random.choice(p=prob)
        cur_aoi_id = aoi_df.iloc[sampled_idx]['aoi_id']

        return {
            'time': cur_time,
            'act': cur_act,
            'aoi_id': cur_aoi_id
        }

    def _pref_return(self, act, current_traj):
        cur_time = self._time_sample(act, current_traj[-1]['time'])
        candidates = [rcd['aoi_id'] for rcd in current_traj if rcd['act'] == act]
        if len(candidates) == 0:
            return None
        aoi_id = random.choice(candidates)
        return {
            'time': cur_time,
            'act': act,
            'aoi_id': aoi_id
        }

    def _choose_pattern(self, current_location):
        locations = {rcd['aoi_id'] for rcd in current_location}
        visited_loc_num = len(locations)
        p_new = np.random.uniform(0, 1)

        if ((p_new <= self.rho * math.pow(visited_loc_num, -self.gamma) and visited_loc_num < self.total_loc_num)
                or visited_loc_num == 1):
            return 'explore'
        else:
            return 'return'

    def generate(self, act_traj: List):
        """
        :param act_traj:
        :return: [{'time', 'aoi_id', 'act_type', 'act_id', }]
        """
        gen_traj = [self._choose_start(act_traj[0])]
        for idx, act in enumerate(act_traj[1:]):
            pattern = self._choose_pattern(gen_traj)
            if pattern == 'explore':
                new_rcd = self._explore(act, gen_traj)
                if new_rcd is None:
                    new_rcd = self._pref_return(act, gen_traj)
            else:
                assert pattern == 'return'
                new_rcd = self._pref_return(act, gen_traj)
                if new_rcd is None:
                    new_rcd = self._explore(act, gen_traj)
            gen_traj.append(new_rcd)
        gen_df = pd.DataFrame(gen_traj)
        gen_df = gen_df.rename(columns={'time': 'hour', 'act': 'act_id', 'aoi_id': 'aoi_id'})
        return gen_df


def naive_epr_generate(city):
    data_dir = 'cleared_data'
    tmp_df = pd.read_csv(join(data_dir, city, 'grid_region_feature.csv'))
    aoi_gdf = gpd.GeoDataFrame(
        data=tmp_df[[col for col in tmp_df.columns if col != 'geometry']],
        geometry=gpd.GeoSeries.from_wkt(tmp_df['geometry'])
    )

    act_type_num = 10
    feat_cols = [f'act{k}' for k in range(act_type_num)]
    aoi_gdf['poi_count'] = aoi_gdf.apply(lambda row: row[feat_cols].sum(), axis=1)

    year = 2024
    base_time = pd.to_datetime(f'{year}/01/01 00:00:00')
    gen_traj_num = 100
    depr = DensityEPR()
    gen_dfs = []
    for tid in range(gen_traj_num):
        day_sample = random.randint(0, 364)
        hour_sample = random.randint(3, 11)
        start_time = base_time + pd.Timedelta(day_sample, unit='d') + pd.Timedelta(hour_sample, unit='h')
        end_time = base_time + pd.Timedelta(day_sample + 1, unit='d')
        tdf = depr.generate(
            start_time, end_time, aoi_gdf,
            relevance_column='poi_count'
        )
        tdf['traj_id'] = tid
        gen_dfs.append(tdf)
    gen_df = pd.concat(gen_dfs)
    print(gen_df.head().to_string())
    gen_df.to_csv(join(data_dir, city, 'naive_epr_gen.csv'), index=False)


def act_epr_generate(city):
    data_dir = 'cleared_data'
    cfg = yaml.safe_load(open('epr_config.yaml', 'r'))
    act_df = pd.read_csv(join(data_dir, 'act_gen.csv'))
    act_groups = act_df.groupby(by='traj_id')

    tmp_df = pd.read_csv(join(data_dir, city, 'grid_region_feature.csv'))
    aoi_gdf = gpd.GeoDataFrame(
        data=tmp_df[[col for col in tmp_df.columns if col != 'geometry']],
        geometry=gpd.GeoSeries.from_wkt(tmp_df['geometry'])
    )
    feat_cols = [f'act{k}' for k in range(10)]
    for col in feat_cols:
        aoi_gdf[col] = aoi_gdf[col] / aoi_gdf[col].sum()

    act_dur = np.load(join(data_dir, 'act_dur.npy'))
    for i in range(len(act_dur)):
        act_dur[i] = act_dur[i] / act_dur[i].sum()

    model = ActEPR(cfg=cfg, aoi_gdf=aoi_gdf, act_time_distri=act_dur)

    gen_dfs = []
    for traj_id, group in tqdm(act_groups, total=len(act_groups)):
        act_list = act_groups['act_id'].tolist()
        tdf = model.generate(act_list)
        tdf['traj_id'] = traj_id
        gen_dfs.append(tdf)
    gen_df = pd.concat(gen_dfs)
    print(gen_df.head().to_string())
    gen_df.to_csv(join(data_dir, city, 'act_epr_gen.csv'), index=False)


if __name__ == '__main__':
    # naive_epr_generate('nbo')
    act_epr_generate('nbo')
