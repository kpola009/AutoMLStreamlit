import autosklearn.regression

def autoregression(X_train, X_test, y_train, y_test, mintime):

    mintime = int(mintime)

    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=mintime)
    automl.fit(X_train, y_train)

    leadmodels = automl.leaderboard()
    return leadmodels