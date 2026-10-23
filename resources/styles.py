#--------------------             
# Author : Serge Zaugg
# Description : overall page styles (affects all pages) 
#--------------------

import streamlit as st

def apply_global_styles():
    """Apply global CSS styles to the Streamlit app."""

    # enforce min and max accepted screen width via CSS
    st.markdown("""
        <style>
            .block-container {
                max-width: 1500px;
                min-width: 1400px;
                padding-left: 2rem;
                padding-right: 2rem;
                margin-left: auto;
                margin-right: auto;
                box-sizing: border-box;
            }
        </style>
    """, unsafe_allow_html=True)

    # Reduce spacing around st.divider() 
    st.markdown("""
        <style>
            div[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] hr) {
                margin-top: -1.9rem !important;
                margin-bottom: -1.0rem !important;}
        </style>
        """, unsafe_allow_html=True)

    # space between main region and nav bar
    st.markdown("""
        <style>
            [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
            margin-top: -1rem !important;
            padding-left: 1.5rem;
            margin-top: -1rem !important;  /* pull dashboard upward */
            } /* here 
        </style>
        """, unsafe_allow_html=True)

    # height of nav bar 
    st.markdown("""
    <style>
        [data-testid="stHeader"] {
            height: 3rem; 
            min-height: 3rem};
    </style>
    """, unsafe_allow_html=True)


    # All navigation tabs: same fixed width
    st.markdown("""
    <style>
        .stSidebar a {
            width: 150px !important;
            min-width: 135px !important;
            max-width: 150px !important;
            box-sizing: border-box !important;
            justify-content: left !important;
            text-align: left !important;
            white-space: nowrap !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # copyright notice and version at bottom-left
    st.markdown("""
    <style>
        .app-label {
            position: fixed;
            bottom: 15px;
            left: 20px;
            color: #9CA3AF;
            font-size: 15px;
            line-height: 1.4;
            z-index: 999999;
            pointer-events: none;
        }
    </style>
    <div class="app-label">
        v1.2.0<br>© 2026 Serge Zaugg
    </div>
    """, unsafe_allow_html=True)

  



   



   
    