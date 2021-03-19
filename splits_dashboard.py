from splits import *

import streamlit as st
import streamlit.components.v1 as components
PAGE_CONFIG = {"page_title":"Covid-19 dashboard","page_icon":":mask:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    st.title("Splits analysis")
    runner_name = st.text_input("Enter runner name:", 'marco')
    runner = Runner(runner_name, alpha=.05, q=.95)
    split_list = list(runner.split_map.split_code.values)
    #chosen_split = st.sidebar.radio("Split:",('Cases', 'Deaths', 'Reproduction rate', 'Positive rate'), index = 0)
    chosen_split = st.selectbox('Choose split:', split_list, default = [split_list[0]])
    

    fig_violin = runner.boxplot(points = "outliers")
    st.write(fig_violin)
    #runner.plot_splits_over_time('M', q=.05)
    #runner.plot_resets()

    sa = runner.split_analysis('average_run')

    #split= "Palace Done"
    #current_time = process_time("2:43:47")
    #res = runner.predict(split, current_time, endsplit=None,display=False, verbose=True)
    #nice_time(res['hpd_median'])
    
if __name__ == '__main__':
    main()
