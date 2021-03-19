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

def main():
            
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_runner(runner_name):
        return Runner(runner_name, alpha=alpha, q=.95)

    st.title("Splits analysis")
    runner_name = st.sidebar.text_input("Runner name", 'marco')
    st.sidebar.write("Runner needs to have uploaded splits to splits.io")
    try:
        runner = get_runner(runner_name)
    except:
        st.error("Could not get runner's splits")
    split_list = list(runner.split_map.split_code)
    
    # Last split
    chosen_split_code = st.selectbox('Last split', split_list, index=0)
    chosen_split_id = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_id"].values[0]
    chosen_split_name = runner.split_map.loc[runner.split_map.split_code == chosen_split_code,"split_name"].values[0]

    # Last split time
    current_time = st.text_input(f"{chosen_split_name} end time", "00:00:00")
    if current_time != "":
        st.write(current_time)
        pct = process_time(current_time)
    
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
    st.write(runner.plot_splits_over_time('M', q=.05)) # Split improvement over time
    st.write(runner.plot_resets()) # Number of resets
    st.write(runner.split_analysis('average_run')) # Average run
    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
