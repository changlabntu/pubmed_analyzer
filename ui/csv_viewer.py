# ===========================================
# 📄 ui/csv_viewer.py - CSV Viewer Components
# ===========================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def load_csv_files():
    """Find all CSV files in current directory"""
    return sorted([str(f) for f in Path(".").glob("*.csv")])

def create_file_selector():
    """Create file upload/selection interface"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload CSV file", 
            type=['csv'],
            help="Upload research paper CSV data"
        )
    
    with col2:
        csv_files = load_csv_files()
        selected_file = st.selectbox(
            "📂 Or select existing:",
            ["None"] + csv_files
        ) if csv_files else "None"
    
    # Return the dataframe and filename
    if uploaded_file:
        return pd.read_csv(uploaded_file), uploaded_file.name
    elif selected_file != "None":
        return pd.read_csv(selected_file), selected_file
    
    return None, None

def create_data_filters(df):
    """Create interactive filters for the dataframe"""
    filtered_df = df.copy()
    
    st.markdown("### 🔍 Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # University filter
        if 'university' in df.columns:
            universities = ['All'] + sorted(df['university'].dropna().unique())
            selected_uni = st.selectbox("🏫 University:", universities)
            if selected_uni != 'All':
                filtered_df = filtered_df[filtered_df['university'] == selected_uni]
    
    with col2:
        # Department filter  
        if 'department' in df.columns:
            departments = ['All'] + sorted(df['department'].dropna().unique())
            selected_dept = st.selectbox("🏛️ Department:", departments)
            if selected_dept != 'All':
                filtered_df = filtered_df[filtered_df['department'] == selected_dept]
    
    with col3:
        # Year filter (if you add publication years)
        if 'year' in df.columns:
            years = sorted(df['year'].dropna().unique())
            if years:
                year_range = st.slider(
                    "📅 Publication Year Range:",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years)))
                )
                filtered_df = filtered_df[
                    (filtered_df['year'] >= year_range[0]) & 
                    (filtered_df['year'] <= year_range[1])
                ]
    
    # Text search
    search_term = st.text_input("🔎 Search in titles/abstracts:")
    if search_term:
        mask = pd.Series([False] * len(filtered_df))
        if 'name' in filtered_df.columns:
            mask |= filtered_df['name'].str.contains(search_term, case=False, na=False)
        if 'abstract' in filtered_df.columns:
            mask |= filtered_df['abstract'].str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    return filtered_df

def display_data_table(df, max_rows="All"):
    """Display the data table with formatting and selection"""
    
    display_df = df.copy()
    
    # Always remove abstract from table display
    if 'abstract' in display_df.columns:
        display_df = display_df.drop('abstract', axis=1)
    
    # Limit rows
    if max_rows != "All":
        display_df = display_df.head(max_rows)
    
    # Configure column widths and text wrapping
    column_config = {
        "name": st.column_config.TextColumn(
            "📝 Title", 
            width="large",
            help="Full paper title"
        ),
        "PMID": st.column_config.TextColumn("🆔 PMID", width="small"),
        "corresponding_author": st.column_config.TextColumn("👤 Author", width="medium"),
        "university": st.column_config.TextColumn("🏫 University", width="large"),
        "department": st.column_config.TextColumn("🏛️ Department", width="medium")
    }
    
    # Display with nice formatting and selection enabled
    selected_data = st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Display abstract for selected row
    if selected_data.selection.rows:
        selected_idx = selected_data.selection.rows[0]
        if selected_idx < len(df):
            selected_paper = df.iloc[selected_idx]
            
            st.markdown("### 📄 Abstract")
            st.markdown("---")
            
            # Paper info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📝 Title:** {selected_paper.get('name', 'N/A')}")
                st.markdown(f"**👤 Author:** {selected_paper.get('corresponding_author', 'N/A')}")
            with col2:
                st.markdown(f"**🆔 PMID:** {selected_paper.get('PMID', 'N/A')}")
                st.markdown(f"**🏫 University:** {selected_paper.get('university', 'N/A')}")
            
            # Abstract content
            abstract = selected_paper.get('abstract', 'N/A')
            if abstract and abstract != 'N/A':
                st.markdown("**Abstract:**")
                st.markdown(f"_{abstract}_")
            else:
                st.info("No abstract available for this paper.")
    else:
        st.info("👆 Select a row above to view the abstract")

def create_summary_metrics(df):
    """Create summary metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Papers", len(df))
    
    with col2:
        if 'university' in df.columns:
            unique_unis = df['university'].nunique()
            st.metric("🏫 Universities", unique_unis)
    
    with col3:
        if 'department' in df.columns:
            unique_depts = df['department'].nunique()
            st.metric("🏛️ Departments", unique_depts)
    
    with col4:
        if 'abstract' in df.columns:
            with_abstracts = len(df[df['abstract'] != 'N/A'])
            st.metric("📝 With Abstracts", with_abstracts)