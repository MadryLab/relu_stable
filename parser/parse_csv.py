import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Pass in csv to parse')
parser.add_argument('--csv_name', dest='csv_name')
args = parser.parse_args()

filename = [args.csv_name]

mnist_test_labels = pd.read_csv("parser/mnist_test_labels.csv").set_index("SampleNumber")

def preprocess_summary_file(dt, mnist_test_labels, get_last=False):
    dt.SampleNumber -=1
    dt.PredictedIndex -=1
    dt = dt.drop_duplicates(
        subset = "SampleNumber", keep= "last" if get_last else "first"
    ).set_index("SampleNumber").sort_index().join(mnist_test_labels)
    dt["IsCorrectlyPredicted"] = dt.PredictedIndex == dt.TrueIndex
    dt["ProcessedSolveStatus"] = dt["SolveStatus"].apply(process_solve_status)
    #dt["ProcessedSolveStatusFast"] = (dt["ProcessedSolveStatus"] == 'ProvablyRobust') & (dt["SolveTime"] < 5)
    dt["BuildTime"] = dt["TotalTime"] - dt["SolveTime"]
    return dt

def get_dt(filename):
    dt = pd.read_csv(filename)
    return preprocess_summary_file(dt, mnist_test_labels)

def process_solve_status(s):
    if s == "InfeasibleOrUnbounded":
        return "ProvablyRobust"
    elif s == "UserLimit":
        return "StatusUnknown"
    else:
        return "Vulnerable"

def summarize_processed_solve_status(filename):
    dt = get_dt(filename)
    #return dt.groupby("ProcessedSolveStatusFast").TotalTime.count().rename(filename)
    return dt.groupby("ProcessedSolveStatus").TotalTime.count().rename(filename)

def summarize_time(filename, agg_by="mean", correct_only=True, exclude_timeouts=False, exclude_skipped_natural_incorrect=True):
    dt = get_dt(filename)
    if correct_only:
        dt = dt[dt.IsCorrectlyPredicted]
    if exclude_timeouts:
        dt = dt[dt["ProcessedSolveStatus"]!="StatusUnknown"]
    if exclude_skipped_natural_incorrect:
        dt = dt[dt["SolveTime"]!=0]
    return dt[["BuildTime", "SolveTime", "TotalTime"]].agg(agg_by).rename(filename)

def summarize_time_stdev(filename, agg_by="std", correct_only=True, exclude_timeouts=False, exclude_skipped_natural_incorrect=True):
    dt = get_dt(filename)
    if correct_only:
        dt = dt[dt.IsCorrectlyPredicted]
    if exclude_timeouts:
        dt = dt[dt["ProcessedSolveStatus"]!="StatusUnknown"]
    if exclude_skipped_natural_incorrect:
        dt = dt[dt["SolveTime"]!=0]
    return dt[["BuildTime", "SolveTime", "TotalTime"]].agg(agg_by).rename(filename)

def summarize_accuracy(filename):
    dt = get_dt(filename)
    return dt.groupby("IsCorrectlyPredicted").TotalTime.count().rename(filename)/len(dt)

def get_summary(f, filenames):
    return pd.concat(map(f, filenames), axis=1).transpose()


# Main code
print(get_summary(summarize_processed_solve_status, filename))
print(get_summary(summarize_time, filename))