import streamlit as st
from ui.csv_viewer import (
    create_file_selector, 
    create_data_filters, 
    display_data_table,
    create_summary_metrics
)
from ui.visualizations import (
    create_university_chart,
    create_department_chart
)

# Configure page
st.set_page_config(
    page_title="PubMed Research Tool",
    page_icon="🧬", 
    layout="wide"
)

def main():
    """Main application with clean organization"""
    
    # App header
    st.title("🧬 PubMed Research Analysis Tool")
    st.markdown("Analyze research papers with AI-powered insights")
    
    # Navigation
    tab1, tab2 = st.tabs(["🔍 Search PubMed", "📊 Analyze CSV Data"])
    
    with tab1:
        st.markdown("### Search & Export Papers")
        # Your search interface here
        
    with tab2:
        st.markdown("### CSV Data Analysis")
        
        # File selection
        df, filename = create_file_selector()
        
        if df is not None:
            # Summary metrics
            create_summary_metrics(df)
            
            # Data filtering
            filtered_df = create_data_filters(df)
            
            # Data table
            st.markdown(f"### 📋 Data Table ({len(filtered_df)} papers)")
            
            display_options_col1, display_options_col2 = st.columns([3, 1])
            
            with display_options_col2:
                max_rows = st.selectbox("Rows:", [10, 25, 50, 100, "All"], index=4)  # Default to "All"
            
            # Add custom CSS for better table display
            st.markdown("""
            <style>
            .stDataFrame {
                width: 100%;
            }
            .stDataFrame > div {
                width: 100%;
                overflow-x: auto;
            }
            /* Ensure text wrapping in cells */
            .stDataFrame [data-testid="stDataFrameResizable"] div[data-testid="cell"] {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                max-height: none !important;
                height: auto !important;
                overflow: visible !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            display_data_table(filtered_df, max_rows)
            
            # Visualizations moved to bottom
            st.markdown("### 📊 Visual Analysis")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                create_university_chart(filtered_df)
            
            with viz_col2:
                create_department_chart(filtered_df)
            
            # Download filtered data
            if len(filtered_df) > 0:
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Filtered Data",
                    data=csv_data,
                    file_name=f"filtered_{filename}",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()