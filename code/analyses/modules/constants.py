from pathlib import Path

PATH_DATA = Path("data")
PATH_DATA_FIGURES = Path("data_figures")
PATH_FIGURES = Path("figures")

S_MAX = 5420

experiments = [
    # Intra
    # ("R2_intra/", "R2_intra"),
    # vs col-def
    ("R2_vs_col_def/0_col_4_def/", "0_col_4_def"),
    ("R2_vs_col_def/1_col_3_def/", "1_col_3_def"),
    ("R2_vs_col_def/2_col_2_def/", "2_col_2_def"),
    ("R2_vs_col_def/3_col_1_def/", "3_col_1_def"),
    ("R2_vs_col_def/4_col_0_def/", "4_col_0_def"),
    # vs const
    ("R2_vs_const/4_const1/", "4_const1"),
    ("R2_vs_const/4_const3/", "4_const3"),
    ("R2_vs_const/4_const5/", "4_const5"),
    # vs opt
    ("R2_vs_opt/4_opt/", "4_opt"),
    # vs col-def switch
    ("R2_vs_col_def_switch/0_col_4_def__4_col_0_def/", "0_col_4_def"),
    ("R2_vs_col_def_switch/0_col_4_def__4_col_0_def/", "4_col_0_def"),
    ("R2_vs_col_def_switch/4_col_0_def__0_col_4_def/", "0_col_4_def"),
    ("R2_vs_col_def_switch/4_col_0_def__0_col_4_def/", "4_col_0_def"),
]
