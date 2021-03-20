import numpy as np, pandas as pd
#from pandas.api.types import CategoricalDtype # used to reorder the ridgeplot
from scipy.stats import halfnorm, norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats.kde import gaussian_kde # to add empirical pdf to plotly
#from joypy import joyplot # for ridgeplot
import datetime
from plotly.subplots import make_subplots
import requests
from itertools import cycle
from plotly import colors as pc
pccols = cycle(pc.qualitative.Plotly)

def cond_cdf(x, current_split, current_time, endsplit, mus, sigmasq):
    current_split += 1
    endsplit += 1
    sumu = np.sum(np.array(mus)[current_split:endsplit])
    sqrtsumsig = np.sqrt(np.sum([x**2 for x in np.array(sigmasq)[current_split:endsplit]]))
    return norm.cdf((x-current_time - sumu)/(sqrtsumsig))

def cond_pdf(a, b, current_split, current_time, endsplit, mus, sigmasq):
    return cond_cdf(b, current_split, current_time, endsplit, mus=mus, sigmasq=sigmasq) - cond_cdf(a, current_split, current_time, endsplit, mus=mus, sigmasq=sigmasq)

def cond_ppf(u, current_split, current_time, endsplit, mus, sigmasq):
    current_split += 1
    endsplit += 1
    sumu = np.sum(np.array(mus)[current_split:endsplit])
    sqrtsumsig = np.sqrt(np.sum([x**2 for x in np.array(sigmasq)[current_split:endsplit]]))
    return norm.ppf(u)*sqrtsumsig + current_time + sumu

def nice_time(seconds):
    return str(datetime.timedelta(seconds=round(seconds)))

def process_time(split_time):
    split_time = split_time.split(':')
    split_time = [int(x) for x in split_time]
    if len(split_time) == 3:
        split_time = split_time[0]*3600 + split_time[1]*60 + split_time[2]
    elif len(split_time) == 2:
        split_time = split_time[0]*3600 + split_time[1]*60
    return split_time

def get_game_category(game_id, game_category_id):
    resp = requests.get(f'https://www.speedrun.com/api/v1/games/{game_id}')
    game_name = resp.json()['data']['names']['international']
    resp = requests.get(f'https://www.speedrun.com/api/v1/games/{game_id}/categories')
    df = pd.DataFrame(resp.json()['data'])
    game_category_name = df.loc[df['id'] == game_category_id,'name'].values[0]
    return game_name, game_category_name

def get_game_cover(game_id):
    resp = requests.get(f'https://www.speedrun.com/api/v1/games/{game_id}')
    game_cover_url = resp.json()['data']['assets']['cover-large']['uri']
    return game_cover_url

class Runner:
    def __init__(self, user, alpha=.05, q=1):
        self.user = user
        self.alpha = alpha
        self.q = q
        self.splits = self.get_splits()
        self.split_map = self.make_split_map()
        self.speedrun_category = {'game_code':self.game_id, 'game_category':self.game_category_id, "user":self.user}
        self.game_name, self.game_category_name = get_game_category(self.game_id, self.game_category_id)
        self.clean_s = self.clean_splits()
        #self.tidy_s = self.tidy_splits()
        self.pb_time = self.get_pb()
        self.wr_time = self.get_wr()
        self.target = self.pb_time #to change
        self.targets = {'target':self.target, 'PB':self.pb_time, 'WR':self.wr_time}
        
    def make_split_map(self):
        df = self.splits[['split_id', 'split_name']].groupby(['split_id', 'split_name']).count().reset_index()
        df['split_code'] = [f'{row.split_id} - {row.split_name}' for _, row in df.iterrows()]
        return df.sort_values('split_id')

    def get_pb(self):
        api_pb = f'https://www.speedrun.com/api/v1/users/{self.speedrun_category["user"]}/personal-bests'
        response = requests.get(api_pb)
        return response.json()['data'][0]['run']['times']['primary_t']

    def get_wr(self):
        game_code = self.speedrun_category['game_code']
        game_category = self.speedrun_category['game_category']
        api_wr = f'https://www.speedrun.com/api/v1/leaderboards/{game_code}/category/{game_category}?top=1'
        response = requests.get(api_wr)
        return response.json()['data']['runs'][0]['run']['times']['primary_t']

    def get_splits(self):
        response = requests.get(f"https://www.speedrun.com/api/v1/users/{self.user}/personal-bests")
        if response.status_code != 200:
            print(response.json()['message'])
            raise Exception('Problem with user')
        try:
            api_splits = response.json()['data'][0]['run']['splits']['uri'] # May need to change number of list item e.g. Kfjs: 2
            print(api_splits)
        except:
            print('No split data')
            raise
        self.game_id = response.json()['data'][0]['run']['game']
        self.game_category_id = response.json()['data'][0]['run']['category']
        res_splits = requests.get(api_splits.replace("v3", "v4"), params={'historic':1}) # barbaric way to switch to API v4
        self.sob_time = res_splits.json()['run']['realtime_sum_of_best_ms']/1000

        all_splits = pd.DataFrame(res_splits.json()['run']['segments'])#[0]['histories'])
        attempt_numbers = []
        histories = []
        for i, row in all_splits.iterrows():
            for h in row['histories']:
                if h['attempt_number'] not in attempt_numbers:
                    attempt_numbers.append(h['attempt_number'])
                    histories.append({"id":row['id'], "split_id":0, "split_name":"Run start", "split_code": "0 - Run start", 'attempt_number':h['attempt_number'], 'split_duration':0})
                histories.append({"id":row['id'], "split_id":row['segment_number']+1, "split_name":row['name'], "split_code": f'{row["segment_number"]+1} - {row["name"]}',
                                    'attempt_number':h['attempt_number'], 'split_duration':h['realtime_duration_ms']/1000})
        histories = pd.DataFrame(histories)
        #histories.append(pd.DataFrame({'id':[0], "split_id":[0], "split_name":['Run start'], 'attempt_number':[0], 'split_duration':[0]})) # NEW trying to add 0th split
        attempts = pd.DataFrame(res_splits.json()['run']['histories'])
        self.attempts = attempts
        attempts.drop(['gametime_duration_ms', 'realtime_duration_ms'], axis=1, inplace=True)
        splits_hist = histories.merge(attempts, on='attempt_number')
        #splits_hist['split_code'] = [f'{row.split_id} - {row.split_name}' for _,row in splits_hist.iterrows()]

        return splits_hist

    def clean_splits(self):
        nrows = self.splits.shape[0]

        def thresh_mean(x, q=self.q):
            return np.median(x)
        def thresh_std(x, q=self.q):
            return np.median(np.abs(x-np.median(x)))

        def thresh_mean(x, q=self.q):
            hold = np.quantile(x, self.q)
            threshed = [i for i in x if i < hold]
            return np.median(threshed)
        def thresh_std(x, q=self.q):
            hold = np.quantile(x, self.q)
            threshed = [i for i in x if i < hold]
            return np.median(np.abs(threshed-np.median(threshed)))
        if False:
            def thresh_mean(x, q=self.q):
                hold = np.quantile(x, self.q)
                threshed = [i for i in x if i < hold]
                return np.mean(threshed)
            def thresh_std(x, q=self.q):
                hold = np.quantile(x, self.q)
                threshed = [i for i in x if i < hold]
                return np.std(threshed)

        #splits_agg = pd.DataFrame([{'count':len([i for i in x if i != 0]), 'average':np.mean([i for i in x if i != 0]), 'stdev':np.std([i for i in x if i != 0])} for x in splits_hist.history]) # Removing 0 (probably skips)
        df = self.splits.loc[(self.splits.split_duration > 0) | (self.splits.split_id == 0),:] # remove zeros except for run start        
        df['started_at'] = df['started_at'].apply(pd.Timestamp)
        df = df[df['started_at'] > pd.Timestamp.today(tz="UTC") + pd.Timedelta('-90 days')] # Keeping latest runs only
        filtered = pd.DataFrame(columns=df.columns)
        for segment in df.split_id.unique():
            dfs = df.loc[df['split_id'] == segment,:]
            hold = np.quantile(dfs['split_duration'], self.q)
            filtered = filtered.append(dfs.loc[dfs['split_duration'] <= hold,:])
        filtered.append(df.loc[df.split_id == 0,:]) # trying to bring back split 0
        self.splits_filtered = filtered.sort_values('split_id')
        #df = df.groupby('split_id').agg({'split_duration':[thresh_mean, thresh_std]})
        df = filtered.groupby('split_id').agg({'split_duration':[thresh_mean, thresh_std]})
        df.columns = df.columns.droplevel(0)
        df.reset_index(inplace=True)
        df.rename(columns={'thresh_mean':'mus', 'thresh_std':'sigmasq'}, inplace = True)   
        df.loc[df["split_id"] == 0, ["mus"]] = 0
        df.loc[df["split_id"] == 0, ["sigmasq"]] = 0
        #empirical_splits = empirical_splits.merge(df, on='split_id')   
        empirical_splits = df
        empirical_splits['game_id'] = self.game_id
        empirical_splits['game_category_id'] = self.game_category_id
        #empirical_splits['split_duration'] = self.splits['duration']
        #empirical_splits['run_time'] = np.cumsum(empirical_splits['split_duration'], axis=0)
        empirical_splits.dropna(inplace=True)
        empirical_splits['average_run'] = np.cumsum(empirical_splits['mus'],axis=0)
        return empirical_splits

    def get_split(self, split):
        if isinstance(split, str):
            split = self.split_map.loc[self.split_map['split_name']==split, 'split_id'].values[0]
        split_name = self.split_map.loc[self.split_map['split_id']==split, 'split_name'].values[0]
        return split, split_name

    def plot_split(self, split):
        time_scale = 60*60
        split, split_name = self.get_split(split)
        # Empirical pdf
        edf = self.tidy_s['split_duration'][self.tidy_s['split'] == split].values
        kde = gaussian_kde(edf)
        x_range = np.linspace(min(edf), max(edf), len(edf))
        empirical_pdf = pd.DataFrame({'x_range': x_range, 'x_kde': kde(x_range)})
        
        
        if False:
            mus = self.clean_s.mus
            sigmasq = self.clean_s.sigmasq
            average_time = np.sum(mus[split:endsplit])
            sqrtsumsq = np.sqrt(np.sum([x**2 for x in sigmasq[split:endsplit]]))
            lam = 5
            #target_range = np.linspace(average_time*(1-lam), average_time*(1+lam)) # Probably going to cause problems
        target_range = x_range#np.linspace(average_time-lam*sqrtsumsq, average_time+lam*sqrtsumsq) # Probably going to cause problems

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        #fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=empirical_pdf['x_kde'], mode='lines',name=f"{split} - {split_name}", fill="tozeroy"))

        if False:
            # HPD
            fig.add_shape(type="line",
                x0=hpd_low/time_scale, y0=0, 
                x1=hpd_high/time_scale, y1=0,
                line=dict(color="Black",width=3))
            
            if endsplit == self.clean_s.shape[0]-1:
                # Target vline
                fig.add_shape(type="line",
                    x0=target/time_scale, y0=-np.max(y)*.05, 
                    x1=target/time_scale, y1=np.max(y)*.05,
                    line=dict(color="Gold",width=3))

        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Density", height=500, template="plotly_white",
        #legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01)
            #legend={"orientation":'h'}
        )
        fig.show()
        return fig


    def predict(self, split, time, endsplit=None, display=False, verbose = False):
        split, split_name = self.get_split(split)        
        if endsplit is None:
            endsplit = np.max(self.split_map.split_id)
        endsplit, endsplit_name = self.get_split(endsplit)
        
        mus = self.clean_s.mus
        sigmasq = self.clean_s.sigmasq
        
        targets = self.targets
        target = targets['target']
        pb_time = targets['PB']
        wr_time = targets['WR']

        p_target  = cond_cdf(target, split, time, endsplit, mus=mus, sigmasq = sigmasq)
        p_pb      = cond_cdf(pb_time, split, time, endsplit, mus=mus, sigmasq = sigmasq)
        p_wr      = cond_cdf(wr_time, split, time, endsplit, mus=mus, sigmasq = sigmasq)
        hpd_low   = cond_ppf(self.alpha/2, split, time, endsplit, mus=mus, sigmasq = sigmasq)
        hpd_median= cond_ppf(.5, split, time, endsplit, mus=mus, sigmasq = sigmasq)
        hpd_high  = cond_ppf(1-self.alpha/2, split, time, endsplit, mus=mus, sigmasq = sigmasq)

        target_range = np.linspace(hpd_low*(1-0.005), hpd_high*(1+0.005), num=100)
        #print(target_range)

        if hpd_median < 60:
            time_scale = 1
            time_scale_name = "s"
        elif hpd_median < 60*60:
            time_scale = 60
            time_scale_name = "min"
        else:
            time_scale = 60*60
            time_scale_name = "h"


        if verbose:
            print(f'Previous split: \t{split} - {split_name} \t=== \t Split end time:\t{nice_time(time)}')
            print(f'Target split: \t{endsplit} - {endsplit_name} \t===\tExpected end time:\t{nice_time(hpd_median)}')
            print(f'{(1-self.alpha)*100:.0f}% chance of arriving between {nice_time(hpd_low)} and {nice_time(hpd_high)}')
            print(f'Probability of doing better than the target ({nice_time(target)}): {p_target*100:.0f}%')
            print(f'Probability of getting a PB ({nice_time(pb_time)}): {p_pb*100:.2f}%')
            print(f'Probability of getting the WR ({nice_time(wr_time)}): {p_wr*100:.2f}%')

        if display:
            # Theoretical pdf
            x = target_range[1:]-(target_range[1]-target_range[0])/2
            pdf = np.diff([cond_cdf(x=x, current_split=split, current_time=time, endsplit=endsplit, mus=mus, sigmasq=sigmasq) for x in target_range])/(target_range[1]-target_range[0])
            y = pdf/np.sum(pdf)

            # Create figure with secondary y-axis
            #fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x/time_scale, y=y, mode='lines',name="Theoretical pdf", fill="tozeroy"))

            # HPD
            fig.add_shape(type="line",
                x0=hpd_low/time_scale, y0=0, 
                x1=hpd_high/time_scale, y1=0,
                line=dict(color="Black",width=3))
            
            if endsplit == np.max(self.split_map.split_id):
                # Target vline
                fig.add_shape(type="line",
                    x0=target/time_scale, y0=-np.max(y)*.05, 
                    x1=target/time_scale, y1=np.max(y)*.05,
                    line=dict(color="Gold",width=3))

            fig.update_layout(xaxis_title=f"Time ({time_scale_name})", yaxis_title="Density", height=500, template="plotly_white",
                              )

            #fig.update_layout(xaxis = dict( tickmode = 'array',tickvals = [1, 3, 5, 7, 9, 11], ticktext = ['One', 'Three', 'Five', 'Seven', 'Nine', 'Eleven']))
            
            fig.show()
        return {'split_id':split, 'split_name':split_name, 'endsplit_id':endsplit, 'endsplit_name':endsplit_name, 'time':time, 
                'hpd_low':hpd_low, 'hpd_median':hpd_median, 'hpd_high':hpd_high, 
                'p_target':p_target, 'p_pb':p_pb, 'p_wr':p_wr}

    def split_analysis(self, run_col):
        """
        Expecting a dataframe containing split_id, split_duration, mus, sigmasq
        """
        time_scale=60*60
        df_run = self.clean_s
        targets = self.targets
        target = targets['target']
        pb_time = targets['PB']
        wr_time = targets['WR']
        res = []
        for i, row in df_run.iterrows():
            res.append(self.predict(row.split_id, row[run_col], endsplit=np.max(self.split_map.split_id), display=False, verbose=False))
        run_split_analysis = pd.DataFrame(res)
        run_split_analysis['nice_time'] = run_split_analysis.time.apply(nice_time)
        run_split_analysis['nice_hpd_low'] = run_split_analysis.hpd_low.apply(nice_time)
        run_split_analysis['nice_hpd_high'] = run_split_analysis.hpd_high.apply(nice_time)
        run_split_analysis = run_split_analysis.merge(df_run)
        run_split_analysis['display_name'] = [f'{row.split_id} - {row.split_name}' for _,row in run_split_analysis.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.p_target, mode='lines', name="Target", marker_color="Black"))
        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.p_pb, mode='lines', name="PB", marker_color="Green"))
        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.p_wr, mode='lines', name="WR", marker_color="Gold"))
        fig.show()

        linecol = '#1f77b4'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.hpd_low/time_scale,
            fill=None,
            mode='lines',
            line_color=linecol,
            name = "",
            #hovertemplate=hover(run_split_analysis, 'ok', 'views'),
            ))
        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.hpd_high/time_scale,
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines', line_color=linecol))

        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=run_split_analysis.hpd_median/time_scale,
            mode='lines', line_color=linecol))

        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=[target/time_scale]*len(run_split_analysis.nice_time),
            mode='lines', line_color='Black'))

        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=[pb_time/time_scale]*len(run_split_analysis.nice_time),
            mode='lines', line_color='Green'))

        fig.add_trace(go.Scatter(x=run_split_analysis.display_name, y=[wr_time/time_scale]*len(run_split_analysis.nice_time),
            mode='lines', line_color='Gold'))
        fig.show()

        return fig

    def boxplot(self, points="outliers"):
        """Points can be all, outliers, suspectedoutliers"""
        df_sh = self.splits_filtered
        time_scale = 60
        fig = go.Figure()
        pointpos = [-0.9,-1.1,-0.6,-0.3]
        for split in df_sh.split_id.unique():
            split, split_name = self.get_split(split)
            ddf = df_sh[df_sh.split_id == split]
            fig.add_trace(go.Violin(
                y=np.array(ddf['split_duration'])/time_scale,
                name=f'{split} - {split_name}',
                pointpos=0
                #name='kale',
        #        marker_color='#3D9970'
            ))

        fig.update_traces(meanline_visible=True,
                  points=points, # show all points
                  jitter=0.5,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count
        fig.update_layout(
            title_text="Split times distribution",
            violingap=0, violingroupgap=0, violinmode='overlay')
        fig.show()
        return fig

    def plot_splits_over_time(self, freq, q=.1):
        """e.g. freq can be M or W-MON"""
        def low(x):
            return np.quantile(x, q)
        def high(x):
            return np.quantile(x, 1-q)
        df = self.splits
        df['date'] = pd.to_datetime(df['started_at'])
        dfd = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus=np.median).reset_index().sort_values('date')
        dfd['sigmasq'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(sigmasq=np.std).reset_index()['sigmasq']#.sort_values('date')
        dfd['mus_low'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_low=low).reset_index()['mus_low']
        dfd['mus_high'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_high=high).reset_index()['mus_high']

        time_scale = 60
        fig = go.Figure()
        for split_id in np.sort(dfd.split_id.unique()):
            dfds = dfd[dfd.split_id == split_id]
            split_col = next(pccols)
            name = self.get_split(split_id)
            name = f'{name[0]} - {name[1]}'
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_low']/time_scale, name=name, legendgroup=name,
                                    fill=None, showlegend=None, line_color=split_col,
                                    mode='lines'))
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_high']/time_scale, name = name, legendgroup=name,
                                    fill="tonexty", mode="lines", line_color=split_col))
            #fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus'], name = name, legendgroup=name,mode="lines"))
        fig.show()
        return fig

    def plot_resets(self):
        df = self.splits_filtered[['split_id', 'split_name', 'split_duration']]
        resets = df.groupby(['split_id',"split_name"]).count().reset_index()
        resets.rename(columns={'split_duration':'n_attempts'}, inplace=True)
        n_completed_runs = resets['n_attempts'][resets['split_id'] == np.max(resets['split_id'])].values[0]
        resets['p_continue'] = np.array(list(resets['n_attempts'][1:])+[resets['n_attempts'].iloc[-1]])/resets['n_attempts']
        resets['p_finish_run'] = n_completed_runs/resets['n_attempts']#/np.max(resets['n_attempts'])
        resets['display_name'] = [f'{row.split_id} - {row.split_name}' for _,row in resets.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resets['display_name'], y=resets['p_finish_run'], name="p_finish_run"))
        fig.add_trace(go.Scatter(x=resets['display_name'], y=resets['p_continue'], name="p_contiune"))
        fig.show()
        return fig
    
    def plot_future_splits(self, split="", current_time=""):
        res = []
        for endsplit in np.arange(0, self.split_map['split_id'].iloc[-1]+1):
            res.append(self.predict(0, 0, endsplit)) # Issue here, current_time should be time at end of split 0, not 0
        res = pd.DataFrame(res)
        res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
        res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Blue', text=res['text'], hoverinfo='text', mode='lines'))
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Blue', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
        #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))

        if split != "" and current_time != "":
            res = []
            for endsplit in np.arange(split, self.split_map['split_id'].iloc[-1]+1):
                res.append(self.predict(split, current_time, endsplit))
            res = pd.DataFrame(res)
            res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
            res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]

            fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Gold', text=res['text'], hoverinfo='text', mode='lines'))
            fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Gold', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
            #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))
        fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Likely seconds range")
        return fig

