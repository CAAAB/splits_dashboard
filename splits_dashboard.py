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
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

alpha = .05
def print_prediction(res):
    return f'{100*(1-alpha):.0f}% chance of ending {res["endsplit_name"]} between {nice_time(res["hpd_low"])} and {nice_time(res["hpd_high"])}'

def proba_pb(runner, split_id, current_time):
    return runner.predict(split_id, current_time)['p_pb']

def plot_splits_over_time(runner, freq, bands=False, q=.1):
    """e.g. freq can be M or W-MON"""
    def low(x):
        return np.quantile(x, q)
    def high(x):
        return np.quantile(x, 1-q)
    df = runner.splits.loc[runner.splits.split_duration >0,:]
    df['date'] = pd.to_datetime(df['started_at'])
    
    if bands:
        dfd = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus=np.median).reset_index().sort_values('date')
        dfd['sigmasq'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(sigmasq=np.std).reset_index()['sigmasq']#.sort_values('date')
        dfd['mus_low'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_low=low).reset_index()['mus_low']
        dfd['mus_high'] = df.groupby(['split_id', pd.Grouper(key='date', freq=freq)])['split_duration'].agg(mus_high=high).reset_index()['mus_high']

    time_scale = 60
    fig = go.Figure()
    for split_id in runner.split_map.split_id:
        dfm = df.loc[df.split_id == split_id,:]
        split_col = next(pccols)
        name = runner.get_split(split_id)
        name = f'{name[0]} - {name[1]}'
        sl1 = not bands
        fig.add_trace(go.Scatter(x=dfm['started_at'], y=dfm['split_duration']/time_scale, mode='markers', marker_size=3, name=name,marker_color=split_col, legendgroup=name, showlegend=not bands))
        if bands:
            dfds = dfd[dfd.split_id == split_id]
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_low']/time_scale, name=name, legendgroup=name,
                                    fill=None, showlegend=False, line_color=split_col,
                                    mode='lines'))
            fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus_high']/time_scale, name = name, legendgroup=name,
                                    fill="tonexty", mode="lines", line_color=split_col, showlegend=bands))
        #fig.add_trace(go.Scatter(x=dfds['date'], y=dfds['mus'], name = name, legendgroup=name,mode="lines"))
    fig.update_layout(template="plotly_white", yaxis_title="Split duration (min)")
    fig.show()
    return fig


def main():
            
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_runner(runner_name):
        return Runner(runner_name, alpha=alpha, q=.95)

    st.title("Splits analysis")
    st.text("Enter the run duration until the latest split to get an estimated time of completion for the following segments")
    runner_name = st.sidebar.text_input("Runner name", 'marco')
    st.sidebar.write("Runner needs to have uploaded splits to splits.io")
    try:
        runner = get_runner(runner_name)
    except:
        st.error("Could not get runner's splits")
    st.sidebar.write(runner.game_name)
    st.sidebar.write(runner.game_category_name)
    st.sidebar.markdown(f"![Game cover]({get_game_cover(runner.game_id)})")
    split_list = list(runner.split_map.split_code)
    
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
    st.write(runner.plot_future_splits(split=chosen_split_id, current_time=pct))

    
    # Past splits stats
    fig_violin = runner.boxplot(points = "outliers")
    st.write(fig_violin)
    
    stbands = st.radio("Show bands", ["Yes", "No"], index=1)
    bands = True if stbands == "Yes" else False      
    st.write(plot_splits_over_time(runner, 'M', bands=bands, q=.05)) # Split improvement over time # class method deprecated
    #st.write(runner.plot_resets()) # Number of resets
    #st.write(runner.split_analysis('average_run')) # Average run
    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
