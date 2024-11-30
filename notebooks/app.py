import streamlit as st
import pandas as pd
import numpy as np


# Placeholder to fetch a matrix based on user inputs
def fetch_matrix(region_code, matrix_name, sector_name):
    return pd.DataFrame(
        np.random.rand(5, 5),
        columns=[f"Col_{i+1}" for i in range(5)],
        index=[f"Row_{i+1}" for i in range(5)],
    )


# Placeholder for the balancing algorithm
def balancing_algorithm(matrix, row_sums, col_sums):
    balanced_matrix = matrix.copy()
    balanced_matrix = balanced_matrix.div(balanced_matrix.sum(axis=1), axis=0).mul(row_sums, axis=0)
    balanced_matrix = balanced_matrix.div(balanced_matrix.sum(axis=0), axis=1).mul(col_sums, axis=1)
    return balanced_matrix.fillna(0)


# Initialize session state
if "matrix" not in st.session_state:
    st.session_state.matrix = None
if "edited_matrix" not in st.session_state:
    st.session_state.edited_matrix = None
if "edited_row_sums" not in st.session_state:
    st.session_state.edited_row_sums = None
if "edited_col_sums" not in st.session_state:
    st.session_state.edited_col_sums = None
if "highlight_matrix" not in st.session_state:
    st.session_state.highlight_matrix = None
if "highlight_row_sums" not in st.session_state:
    st.session_state.highlight_row_sums = None
if "highlight_col_sums" not in st.session_state:
    st.session_state.highlight_col_sums = None

# App title and layout
st.title("Matrix Balancer")

# Sidebar for input selection
st.sidebar.header("Matrix Selection")
matrix_name = st.sidebar.selectbox("Select Matrix Name", ["Matrix_A", "Matrix_B", "Matrix_C"])
if matrix_name:
    sector_name = st.sidebar.selectbox("Select Sector Name", ["Sector_X", "Sector_Y", "Sector_Z"])
    region_code = st.sidebar.selectbox(
        "Select Region Code (optional)", ["Region_1", "Region_2", "Region_3", None], index=3
    )

# Load matrix button
if st.sidebar.button("Load Matrix"):
    matrix = fetch_matrix(region_code, matrix_name, sector_name)
    st.session_state.matrix = matrix
    st.session_state.edited_matrix = matrix.copy()
    st.session_state.edited_row_sums = matrix.sum(axis=1).rename("Row Sum")
    st.session_state.edited_col_sums = matrix.sum(axis=0).rename("Column Sum")
    st.session_state.highlight_matrix = pd.DataFrame(
        False, index=matrix.index, columns=matrix.columns
    )
    st.session_state.highlight_row_sums = pd.Series(False, index=matrix.index)
    st.session_state.highlight_col_sums = pd.Series(False, index=matrix.columns)

# If matrix is loaded
if st.session_state.matrix is not None:
    st.subheader("Matrix Editor")
    st.info(
        "Edit the matrix, row sums, and column sums directly. Changes will be highlighted until the balancing algorithm is applied."  # noqa: E501
    )

    col1, col2 = st.columns([4, 1], gap="small")

    def update_highlight(data, original, highlight):
        changes = data != original
        return highlight | changes

    with col1:
        # Editable matrix
        st.markdown("### Matrix Table")
        edited_matrix = st.data_editor(
            st.session_state.edited_matrix,
            use_container_width=True,
            key="matrix_editor",
        )
        st.session_state.highlight_matrix = update_highlight(
            edited_matrix, st.session_state.edited_matrix, st.session_state.highlight_matrix
        )
        st.session_state.edited_matrix = pd.DataFrame(edited_matrix)

        # Editable column sums
        st.markdown("### Column Sums")
        edited_col_sums = st.data_editor(
            pd.DataFrame(st.session_state.edited_col_sums).T,
            use_container_width=True,
            key="col_sum_editor",
        )
        st.session_state.highlight_col_sums = update_highlight(
            pd.Series(edited_col_sums.iloc[0]),
            st.session_state.edited_col_sums,
            st.session_state.highlight_col_sums,
        )
        st.session_state.edited_col_sums = pd.Series(
            edited_col_sums.iloc[0], index=st.session_state.edited_matrix.columns
        )

    with col2:
        # Editable row sums
        st.markdown("### Row Sums")
        # Spacer removed to adjust alignment
        edited_row_sums = st.data_editor(
            pd.DataFrame(st.session_state.edited_row_sums),
            use_container_width=False,  # Keep width aligned
            key="row_sum_editor",
        )
        st.session_state.highlight_row_sums = update_highlight(
            pd.Series(edited_row_sums.iloc[:, 0]),
            st.session_state.edited_row_sums,
            st.session_state.highlight_row_sums,
        )
        st.session_state.edited_row_sums = pd.Series(
            edited_row_sums.iloc[:, 0], index=st.session_state.edited_matrix.index
        )

    # Apply balancing algorithm
    if st.button("Apply Balancing Algorithm"):
        balanced_matrix = balancing_algorithm(
            st.session_state.edited_matrix,
            st.session_state.edited_row_sums,
            st.session_state.edited_col_sums,
        )
        st.session_state.matrix = balanced_matrix
        st.session_state.edited_matrix = balanced_matrix
        st.session_state.edited_row_sums = balanced_matrix.sum(axis=1)
        st.session_state.edited_col_sums = balanced_matrix.sum(axis=0)
        st.session_state.highlight_matrix = pd.DataFrame(
            False, index=balanced_matrix.index, columns=balanced_matrix.columns
        )
        st.session_state.highlight_row_sums = pd.Series(False, index=balanced_matrix.index)
        st.session_state.highlight_col_sums = pd.Series(False, index=balanced_matrix.columns)
        st.success("Balancing algorithm applied and matrix updated!")
