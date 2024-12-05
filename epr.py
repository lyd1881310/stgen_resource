import random

import pandas as pd
import numpy as np
import math
import geopandas as gpd
import skmob
from skmob.models.epr import DensityEPR
from typing import List
from os.path import join


class PrefEPR:
    def __init__(self, cfg, aoi_gdf: gpd.GeoDataFrame, act_time_distri: np.ndarray):
        self.rho = cfg['rho']
        self.gamma = cfg['gamma']
        self.total_loc_num = len(aoi_gdf)
        self.act_time_distri = act_time_distri
        self.aoi_gdf = aoi_gdf

    def _choose_start(self):
        pass

    def _explore(self, act_traj, current_traj):
        cur_idx = len(current_traj)
        cur_act = act_traj[cur_idx]
        dur_distri = self.act_time_distri[cur_act]
        dur = random.sample(dur_distri, 1)[0]



    def _pref_return(self):
        pass

    def _choose_patten(self, current_location):
        locations = {rcd['aoi_id'] for rcd in current_location}
        visited_loc_num = len(locations)
        p_new = np.random.uniform(0, 1)

        if ((p_new <= self.rho * math.pow(visited_loc_num, -self.gamma) and visited_loc_num < self.total_loc_num)
                or visited_loc_num == 1):  # choose to return or explore
            # PREFERENTIAL EXPLORATION
            next_location = self._explore(current_location)
            return next_location

        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return(current_location)
            return next_location

    def generate(self, act_traj: List):
        """
        :param act_traj:
        :return: [{'timestamp', 'aoi_id', 'act_type', 'act_id', }]
        """
        start_rcd = self._choose_start()
        for act in act_traj:
            pass


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


if __name__ == '__main__':
    naive_epr_generate('nbo')

