from splits import *

import streamlit as st
import streamlit.components.v1 as components
PAGE_CONFIG = {"page_title":"Split dashboard","page_icon":":bar_chart:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

alpha = .05
def print_prediction(res):
    return f'{100*(1-alpha):.0f}% chance of ending {res["endsplit_name"]} between {nice_time(res["hpd_low"])} and {nice_time(res["hpd_high"])}'

def proba_pb(runner, split_id, current_time):
    return runner.predict(split_id, current_time)['p_pb']

def find_runner_runs(runner):
    resp = requests.get(f'https://splits.io/api/v4/runners/{runner}/runs')
    temp=pd.DataFrame(resp.json()['runs'])
    temp['game_name'] = [x['name'] for x in temp['game']]
    temp['category_name'] = [x['name'] for x in temp['category']]
    temp['game_cat'] = temp['game_name'] + ' - ' + temp['category_name']
    #print(temp['game_cat'].unique())
    last_runs = temp.sort_values(['game_cat', "created_at"]).groupby('game_cat').last().reset_index()
    return last_runs

def get_run_id(last_runs, game_cat):
    return last_runs.loc[last_runs.game_cat==game_cat,"id"].values[0]

@st.cache()
def plot_splits_over_time(runner, freq, split, bands=False, q=.1):
    """e.g. freq can be M or W-MON"""
    def low(x):
        return np.quantile(x, q)
    def high(x):
        return np.quantile(x, 1-q)
    df = runner.splits.loc[runner.splits.split_duration >0,:]
    if split != "":
        df = df.loc[df.split_id == split,:]
    df['date'] = pd.to_datetime(df['started_at'])
    df['text'] = [f'{row.split_code}<br>{row.date}<br>{nice_time(row.split_duration)}' for _,row in df.iterrows()]
    legend1 = not bands
    legend2 = bands
    legend1 = False
    legend2 = False
    
    if bands:
        dfd = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus=np.median).reset_index().sort_values('date')
        dfd['sigmasq'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(sigmasq=np.std).reset_index()['sigmasq']#.sort_values('date')
        dfd['mus_low'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_low=low).reset_index()['mus_low']
        dfd['mus_high'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_high=high).reset_index()['mus_high']

    time_scale = 60
    fig = go.Figure()
    for split_id in np.unique(df.split_id):
        dfm = df.loc[df.split_id == split_id,:]
        split_col = next(pccols)
        split_name = runner.get_split(split_id)
        name = f'{split_name[0]} - {split_name[1]}'
        fig.add_trace(go.Scatter(x=dfm['started_at'], y=dfm['split_duration']/time_scale, text=dfm['text'], hoverinfo='text', mode='markers', marker_size=3, name=name,marker_color=split_col, legendgroup=name, showlegend=legend1))
        if bands:
            dfds = dfd[dfd.split_id == split_id]
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_low']/time_scale, name=name, legendgroup=name,
                                    fill=None, showlegend=False, line_color=split_col,
                                    mode='lines'))
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_high']/time_scale, name = name, legendgroup=name,
                                    fill="tonexty", mode="lines", line_color=split_col, showlegend=legend2))
        #fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus'], name = name, legendgroup=name,mode="lines"))
    fig.update_layout(template="plotly_white", yaxis_title="Split duration (min)", title=f"Time improvement on {split_name[1]}:")
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()
    return fig

@st.cache()
def plot_expected_run(runner, split="", current_time=""):
    time_scale = 60*60
    time_scale = 1
    res = []
    for endsplit in np.arange(0, runner.split_map['split_id'].iloc[-1]+1):
        res.append(runner.predict(0, 0, endsplit))
    res = pd.DataFrame(res)
    res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
    res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Blue', text=res['text'], hoverinfo='text', mode='lines'))
    fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Blue', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
    #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))
    res0 = res.copy()

    if split != "" and current_time != "":
        res = []
        for endsplit in np.arange(split, runner.split_map['split_id'].iloc[-1]+1):
            res.append(runner.predict(split, current_time, endsplit))
        res = pd.DataFrame(res)
        res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
        res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Gold', text=res['text'], hoverinfo='text', mode='lines'))
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Gold', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
        #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))
    fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Expected time (s)")
    return fig

@st.cache()
def plot_violin(runner, points="outliers"):
    """Points can be all, outliers, suspectedoutliers"""
    df_sh = runner.splits_filtered
    time_scale = 60
    fig = go.Figure()
    pointpos = [-0.9,-1.1,-0.6,-0.3]
    for split in df_sh.split_id.unique():
        split, split_name = runner.get_split(split)
        ddf = df_sh.loc[df_sh.split_id == split,:]
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
        title_text="Historical distribution of each split's duration",
        yaxis_title="Split duration (min)",
        violingap=0, violingroupgap=0, violinmode='overlay', template="plotly_white")
    fig.show()
    return fig

@st.cache()
def plot_pb_vs_best(runner, diff=False):
    df = runner.split_map
    df['text'] = ["Shortest: "+nice_time(x) for x in df.split_shortest_duration]
    fig = go.Figure()
    if diff:
        ytitle = "PB - Best ever (s) "
        fig.add_trace(go.Bar(x = df.split_code, y = df.split_pb - df.split_shortest_duration, text = df.text, hoverinfo="text", name="PB - Best", marker_color="#1f77b4"))
    else:
        ytitle = "Split best time (min)"
        fig.add_trace(go.Bar(x = df.split_code, y = df.split_shortest_duration/60, text = df.text, hoverinfo="text", name="Best ever", marker_color="Gold"))
        fig.add_trace(go.Bar(x = df.split_code, y = df.split_pb/60, text = df.text, hoverinfo="text", name="PB run", marker_color="Green"))
    fig.update_layout(template="plotly_white", yaxis_title=ytitle)
    return fig

def main():
            
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_runner(runner_name, force_splits):
        return Runner(runner_name, force_splits=force_splits,alpha=alpha, q=.95)

    # Sidebar
    st.sidebar.write("Runner needs to have uploaded splits to splits.io")
    runner_name = st.sidebar.text_input("Runner name", 'demon')
    last_runs = find_runner_runs(runner_name)
    game_cat = st.sidebar.selectbox("Category", np.sort(last_runs.game_cat.unique()), index=0)
    force_splits = get_run_id(last_runs, game_cat)
    force_splits = st.sidebar.text_input("Run id", force_splits)
    #force_splits = "7g31"
    try:
        runner = get_runner(runner_name, force_splits)
    except:
        st.error("Could not get runner's splits")
    st.sidebar.write(f"Last uploaded run: {runner.splits.started_at.max()}")
    st.sidebar.write(runner.game_name)
    st.sidebar.write(runner.game_category_name)
    st.sidebar.markdown(f"![Game cover]({get_game_cover(runner.game_id)})")
    split_list = list(runner.split_map.split_code)
    
    st.title(f"{runner_name}'s splits")
    st.markdown("Enter the run duration until the latest split to get an estimated time of completion for the following segments")
    
    # Last split
    chosen_split_code = st.selectbox('Last split', split_list, index=0)
    chosen_split_id = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_id"].values[0]
    chosen_split_name = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_name"].values[0]

    # Last split time
    current_time = st.text_input(f"{chosen_split_name} end time", "00:00:00")
    if current_time != "":
        pct = process_time(current_time)
    st.write(f"{100*proba_pb(runner, chosen_split_id, pct):.0f}% chance of PB ({nice_time(runner.pb_time)})")
    # Target split
    chosen_endsplit_code = st.selectbox('Target split', split_list, index=len(split_list)-1)
    chosen_endsplit_id = runner.split_map.loc[runner.split_map.split_code == chosen_endsplit_code,"split_id"].values[0]
    chosen_endsplit_name = runner.split_map.loc[runner.split_map.split_code == chosen_endsplit_code,"split_name"].values[0]

    # Predict next splits
    res=runner.predict(chosen_split_id, pct, chosen_endsplit_id, display = False, verbose=False)
    st.write(print_prediction(res))
    #st.write(runner.plot_future_splits(split=chosen_split_id, current_time=pct))
    st.markdown('The below graph shows the run time uncertainty at the start of the run (in blue) and at the current split (in yellow) for each of the remaining splits. Hover above the lines to get a credible interval for the end split time')
    st.write(plot_expected_run(runner=runner, split=chosen_split_id, current_time=pct))

    
    # Past splits stats
    st.write(plot_violin(runner, points = "outliers"))
    
    t_split_code = st.selectbox('Split', split_list[1:], index=0)
    t_split_id = runner.split_map.loc[runner.split_map.split_code == t_split_code,"split_id"].values[0]
    t_split_name = runner.split_map.loc[runner.split_map.split_code == t_split_code,"split_name"].values[0]

    stbands = st.radio("Show bands", ["Yes", "No"], index=1)
    bands = True if stbands == "Yes" else False      
    st.write(plot_splits_over_time(runner, 'M', split=t_split_id, bands=bands, q=.05)) # Split improvement over time # class method deprecated
    
    st.subheader("PB to best splits comparison")
    stdiff = st.radio("Show PB and best ever:", ["Side to side", "Difference"], index=0)
    diff = True if stdiff == "Difference" else False
    st.write(plot_pb_vs_best(runner, diff=diff))
    #st.write(runner.plot_resets()) # Number of resets
    #st.write(runner.split_analysis('average_run')) # Average run
    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
