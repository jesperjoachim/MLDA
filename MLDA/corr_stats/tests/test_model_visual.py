import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt

# import seaborn as sns
# import joblib
# import plotly.graph_objects as go

from pandas.testing import assert_frame_equal

from MLDA.miscellaneous.save_and_load_files import load_object, save_object
from MLDA.plot_functions.model_visual import (
    findHighestFeatImportance,
    findCorrBelowLevelBetweenVars,
    screenForCorrLevelAndFeatImp,
    findVarsForPlot,
)


"""Dataset used for test"""


@pytest.fixture()
def dictMLForTest():
    dictML_test = load_object(
        "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
    )
    return dictML_test


@pytest.fixture()
def dictDAForTest():
    dictDA_test = load_object(
        "/home/jesper/Work/MLDA_app/MLDA/jupyter_DA/da_dict_N1.sav"
    )
    return dictDA_test


"""END Dataset used for test"""


"""Testing the function findHighestFeatImportance"""


def test_findHighestFeatImportance_returns_name_of_var_with_the_highest_feature_importanct_when__vars_chosen__is_empty(
    dictMLForTest
):
    # Setup
    desired = "surf_area"

    # Exercise
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    vars_chosen = {}
    actual = findHighestFeatImportance(
        df_feat_imp=df_feat_imp, var_candidates=var_candidates
    )

    # Verify
    assert actual == desired


def test_findHighestFeatImportance_returns_name_of_var_with_the_highest_feature_importanct_when__vars_chosen__has_a_single_item(
    dictMLForTest
):
    # Setup
    desired = "height"

    # Exercise
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    vars_chosen = {"var1": "surf_area"}
    actual = findHighestFeatImportance(
        df_feat_imp=df_feat_imp, var_candidates=var_candidates
    )

    # Verify
    assert actual == desired


def test_findHighestFeatImportance_returns_name_of_var_with_the_highest_feature_importanct_when__vars_chosen__has_two_items(
    dictMLForTest
):
    # Setup
    desired = "roof_area"

    # Exercise
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    var_candidates.remove("height")
    vars_chosen = {}
    actual = findHighestFeatImportance(
        df_feat_imp=df_feat_imp, var_candidates=var_candidates
    )

    # Verify
    assert actual == desired


"""END Testing the function findHighestFeatImportance"""


"""Testing the function findCorrBelowLevelBetweenVars"""


def test_findCorrBelowLevelBetweenVars_returns_list_with_3_vars_that_passes_level_test_when_one_item_in__vars_chosen__(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = ["wall_area", "glazing_area", "glaz_area_distrib"]

    # Exercise
    vars_chosen = {"var1": "surf_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    actual = findCorrBelowLevelBetweenVars(
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.5,
    )

    # Verify
    assert actual == desired


def test_findCorrBelowLevelBetweenVars_returns_list_with_2_vars_that_passes_level_test_when_two_item_in__vars_chosen__(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = ["glazing_area", "glaz_area_distrib"]

    # Exercise
    vars_chosen = {"var1": "surf_area", "var2": "wall_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    var_candidates.remove("wall_area")
    actual = findCorrBelowLevelBetweenVars(
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.5,
    )

    # Verify
    assert actual == desired


def test_findCorrBelowLevelBetweenVars_returns_f_string_which_says_no_candidates_passes_level(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = "var_candidates = ['roof_area', 'height'], vars_chosen_list = ['surf_area', 'wall_area'] has no candidates that passes a level of 0.01"

    # Exercise
    vars_chosen = {"var1": "surf_area", "var2": "wall_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    var_candidates.remove("wall_area")
    var_candidates.remove("glazing_area")
    var_candidates.remove("glaz_area_distrib")
    actual = findCorrBelowLevelBetweenVars(
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.01,
    )

    # Verify
    assert actual == desired


"""END Testing the function findCorrBelowLevelBetweenVars"""


"""Testing the function screenForCorrLevelAndFeatImp"""


def test_screenForCorrLevelAndFeatImp_returns_name_of_highest_feat_imp_value_when_one_item_in__vars_chosen__(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = "glazing_area"

    # Exercise
    vars_chosen = {"var1": "surf_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    actual = screenForCorrLevelAndFeatImp(
        df_feat_imp=df_feat_imp,
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.5,
    )

    # Verify
    assert actual == desired


def test_screenForCorrLevelAndFeatImp_returns_name_of_highest_feat_imp_value_when_two_items_in__vars_chosen__(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = "wall_area"

    # Exercise
    vars_chosen = {"var1": "surf_area", "var2": "glazing_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("surf_area")
    var_candidates.remove("glazing_area")
    actual = screenForCorrLevelAndFeatImp(
        df_feat_imp=df_feat_imp,
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.5,
    )

    # Verify
    assert actual == desired


def test_screenForCorrLevelAndFeatImp_returns_f_string_which_says_no_candidates_passes_level(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = "var_candidates = ['roof_area', 'height'], vars_chosen_list = ['surf_area'] has no candidates that passes a level of 0.01"

    # Exercise
    vars_chosen = {"var1": "surf_area"}
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    var_candidates = xdata_names
    var_candidates.remove("wall_area")
    var_candidates.remove("glazing_area")
    var_candidates.remove("glaz_area_distrib")
    var_candidates.remove("surf_area")
    actual = screenForCorrLevelAndFeatImp(
        df_feat_imp=df_feat_imp,
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=0.01,
    )

    # Verify
    assert actual == desired


"""END Testing the function screenForCorrLevelAndFeatImp"""


"""Testing the function findVarsForPlot"""


def test_findVarsForPlot_returns_dict_with_two_vars_when__num_for_plot__is_2(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = {"var1": "surf_area", "var2": "glazing_area"}

    # Exercise
    num_for_plot = 2
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"]
    actual = findVarsForPlot(
        num_for_plot=num_for_plot,
        xdata_names=xdata_names,
        df_corr=df_corr,
        df_feat_imp=df_feat_imp,
    )

    # Verify
    assert actual == desired


def test_findVarsForPlot_returns_dict_with_two_vars_when__num_for_plot__is_3(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = {"var1": "surf_area", "var2": "glazing_area", "var3": "wall_area"}

    # Exercise
    num_for_plot = 3
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"]
    actual = findVarsForPlot(
        num_for_plot=num_for_plot,
        xdata_names=xdata_names,
        df_corr=df_corr,
        df_feat_imp=df_feat_imp,
    )

    # Verify
    assert actual == desired


def test_findVarsForPlot_returns_f_string_which_says_no_candidates_passes_level(
    dictDAForTest, dictMLForTest
):
    # Setup
    desired = "var_candidates = ['wall_area', 'roof_area'], vars_chosen_list = ['height'] has no candidates that passes a level of 0.1"

    # Exercise
    num_for_plot = 3
    dictDA_test = dictDAForTest
    dictML_test = dictMLForTest
    df_feat_imp = dictML_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["df_feat_imp"]
    df_corr = dictDA_test["correlation"]
    xdata_names = dictML_test["N1"]["xdata_names"].to_list()
    xdata_names.remove("surf_area")
    xdata_names.remove("glazing_area")
    xdata_names.remove("glaz_area_distrib")
    actual = findVarsForPlot(
        num_for_plot=num_for_plot,
        xdata_names=xdata_names,
        df_corr=df_corr,
        df_feat_imp=df_feat_imp,
        level=0.1,
    )

    # Verify
    assert actual == desired


"""END Testing the function findVarsForPlot"""
