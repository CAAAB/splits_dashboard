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

def plot_future_splits(runner, split=None, current_time=None):
        res = []
        for endsplit in np.arange(0, runner.split_map['split_id'].iloc[-1]+1):
            res.append(runner.predict(0, 0, endsplit)) # Issue here, current_time should be time at end of split 0, not 0
        res = pd.DataFrame(res)
        res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
        res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Black', text=res['text'], hoverinfo='text', mode='lines'))
        fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Black', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
        #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))

        if split is not None and current_time is not None:
            res = []
            for endsplit in np.arange(split, runner.split_map['split_id'].iloc[-1]+1):
                res.append(runner.predict(split, current_time, endsplit))
            res = pd.DataFrame(res)
            res['display_name'] = [f'{row.endsplit_id} - {row.endsplit_name}' for _,row in res.iterrows()]
            res['text'] = [f'{row.display_name}<br>High: {nice_time(row.hpd_high)}<br>Median: {nice_time(row.hpd_median)}<br>Low: {nice_time(row.hpd_low)}' for _,row in res.iterrows()]

            fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_low']-res['hpd_median'], line_color='Black', text=res['text'], hoverinfo='text', mode='lines'))
            fig.add_trace(go.Scatter(x=res['display_name'], y=res['hpd_high']-res['hpd_median'], line_color='Black', text=res['text'], hoverinfo='text', fill='tonexty', mode='lines'))
            #fig.add_trace(go.Scatter(x=res['display_name'], y=[0]*res.shape[0], line_color='Black', mode="lines"))
        fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Likely seconds range")
        return fig

def main():
            
    @st.cache(suppress_st_warning=True)
    def get_runner(runner_name):
        return Runner(runner_name, alpha=alpha, q=.95)

    st.title("Splits analysis")
    runner_name = st.sidebar.text_input("Enter runner name:", 'marco')
    runner = get_runner(runner_name)
    split_list = list(runner.split_map.split_code)
    fig_violin = runner.boxplot(points = "outliers")
    st.write(fig_violin)
    
    # Last split
    chosen_split_code = st.selectbox('Last split', split_list, index=0)
    chosen_split_id = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_id"].values[0]
    chosen_split_name = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_name"].values[0]

    # Target split
    current_time = st.text_input(f"{chosen_split_name} end time", "00:00:00")
    pct = process_time(current_time) if current_time is not None else None
    chosen_endsplit_code = st.selectbox('Target split', split_list, index=len(split_list)-1)
    chosen_endsplit_id = runner.split_map.loc[runner.split_map.split_code == chosen_endsplit_code,"split_id"].values[0]
    chosen_endsplit_name = runner.split_map.loc[runner.split_map.split_code == chosen_endsplit_code,"split_name"].values[0]

    st.write(runner.plot_future_splits(split=chosen_split_id, current_time=pct))
    res=runner.predict(chosen_split_id, pct, chosen_endsplit_id, display = False, verbose=False)
    st.write(print_prediction(res))
    st.write(runner.plot_splits_over_time('M', q=.05)) # Split improvement over time
    st.write(runner.plot_resets()) # Number of resets
    st.write(runner.split_analysis('average_run')) # Average run
    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
