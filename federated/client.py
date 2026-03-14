class YieldClient(fl.client.NumPyClient):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="reg:squarederror"
        )
        self.trained = False


    def get_parameters(self, config):
        return []


    def fit(self, parameters, config):

        self.model.fit(self.X, self.y)

        self.trained = True

        preds = self.model.predict(self.X)

        loss = mean_squared_error(self.y, preds)

        return [], len(self.X), {"mse": loss}


    def evaluate(self, parameters, config):

        if not self.trained:
            return 0.0, len(self.X), {"mse": 0.0}

        preds = self.model.predict(self.X)

        loss = mean_squared_error(self.y, preds)

        return loss, len(self.X), {"mse": loss}