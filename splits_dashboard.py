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

def main():
            
    @st.cache(suppress_st_warning=True)
    def get_runner(runner_name):
        return Runner(runner_name, alpha=.05, q=.95)

    st.title("Splits analysis")
    runner_name = st.sidebar.text_input("Enter runner name:", 'marco')
    runner = get_runner(runner_name)
    split_list = list(runner.split_map.split_code)
    fig_violin = runner.boxplot(points = "outliers")
    st.write(fig_violin)
    st.write(runner.plot_splits_over_time('W', q=.05)) # Split improvement over time
    st.write(runner.plot_resets()) # Number of resets    split_list = list(runner.split_map.split_code.values)
    
    chosen_split = st.selectbox('Choose split:', split_list)#, default = [split_list[0]])
    chosen_split_id = runner.split_map.loc[runner.split_map.split_code == chosen_split,"split_id"]
    st.write(chosen_split_id)
    current_time = st.text_input(f"{chosen_split} end time:")
    chosen_endsplit = st.selectbox('Choose target split:', split_list)#, default = [split_list[0]])
    chosen_endsplit_id = runner.split_map.loc[runner.split_map.split_code == chosen_endsplit,"split_id"]
    st.write(chosen_endsplit_id)
    res=runner.predict(chosen_split_id,process_time(current_time), chosen_endsplit_id, display = False, verbose=False)
    st.write(res)
    st.write(runner.plot_splits_over_time('M', q=.05))
    st.write(runner.plot_resets())

    st.write(runner.split_analysis('average_run'))
    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
